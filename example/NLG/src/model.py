#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import sys, os
import time
from collections import OrderedDict
import copy
import math
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parameter import Parameter
import torch.distributed as dist
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

import edge_loralib as lora
from dist_comm_util import *


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def swish(x):
    return x * torch.sigmoid(x)

def restore_compressed_tensor(tensor_to_lora, negative_mask, start_head, end_head, hidden_state_slipt_head_shape):
    current_device_mask = torch.zeros_like(negative_mask, dtype=torch.bool)
    current_device_mask[start_head:end_head] = negative_mask[start_head:end_head]
    restored_tensor = torch.zeros(hidden_state_slipt_head_shape, dtype=tensor_to_lora.dtype, device=tensor_to_lora.device)
    restored_tensor[:, :, current_device_mask[start_head:end_head], :] = tensor_to_lora
    restored_tensor = restored_tensor.view(hidden_state_slipt_head_shape[0], hidden_state_slipt_head_shape[1], -1)
    return restored_tensor


def get_current_hidden_states_shape(tensor_shape, negative_mask, start_head, end_head):
    true_count = torch.sum(negative_mask[start_head:end_head]).item()
    each_device_recv_shape = tensor_shape[:-1] + (true_count, tensor_shape[-1] // len(negative_mask))

    return each_device_recv_shape


def get_current_hidden_states_by_mask(tensor, config, negative_mask, start_head, end_head):
    tensor_shape = tensor.shape
    tensor_splitted = tensor.view(tensor_shape[0], tensor_shape[1], config.n_head, config.n_embd // config.n_head)
    current_device_mask = torch.zeros_like(negative_mask, dtype=torch.bool, device=negative_mask.device)

    current_device_mask[start_head:end_head] = negative_mask[start_head:end_head]
    selected_tensors = tensor_splitted[:, :, current_device_mask, :]
    tensor_to_send = selected_tensors
    return tensor_to_send


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]

        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.head_device_list = config.optimize_tool['head_alloc_device']
        self.current_device_heads = 0
        if dist.get_rank() > 0:
            self.current_device_heads = self.head_device_list[int(dist.get_rank()-1)]
        self.c_attn = lora.MergedLinear(
            nx, n_state * 3,
            r=config.lora_attn_dim,
            client_num=config.client_num,
            lora_alpha=config.lora_attn_alpha,
            lora_dropout=config.lora_dropout,
            enable_lora=[True, False, True],
            fan_in_fan_out=True,
            merge_weights=False,
            n_head=self.n_head,
            current_device_heads=self.current_device_heads
        )
        self.c_proj = Conv1D(n_state, nx)

        self.config = config

    def _attn(self, q, k, v, len_kv=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        if len_kv is not None:
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk = _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10)

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1).contiguous()  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3).contiguous()  # (batch, head, seq_length, head_features)

    def send_combined_attn_grad_to_lora_hook(self, tensor_shape, device):
        def grad_hook(grad):
            dist_send_data(grad, dst=device)

        return grad_hook

    def forward(self, x, history=None, layer_past=None, len_past=None, negative_mask=None):
        hidden_states = x

        time_base_attn_begin = time.time()
        x = self.c_attn(x)

        start_head = 0
        combined_attn_temp_list = []
        combined_attn_temp_sum = None
        query_lora, key_lora, value_lora = 0, 0, 0
        for device_idx, device_num in enumerate(self.config.optimize_tool['head_alloc_device']):
            end_head = start_head + device_num
            current_device_mask = negative_mask[start_head:end_head]
            mask_is_all_False = torch.all(~current_device_mask)
            recv_shape = (*x.shape[:-1], x.shape[-1] // self.n_head * device_num)
            if not mask_is_all_False:
                combined_attn_temp, _ = dist_recv_data(recv_shape, x.device, src=device_idx+1)
                combined_attn_temp.retain_grad()  # ???
                combined_attn_temp.register_hook(self.send_combined_attn_grad_to_lora_hook(combined_attn_temp.shape, device_idx+1))
                
                if combined_attn_temp_sum is None:
                    combined_attn_temp_sum = combined_attn_temp
                else:
                    combined_attn_temp_sum = torch.cat((combined_attn_temp_sum, combined_attn_temp), dim=-1)
            else:
                combined_attn_temp_zero = torch.zeros(recv_shape, device=x.device)
                if combined_attn_temp_sum is None:
                    combined_attn_temp_sum = combined_attn_temp_zero
                else:
                    combined_attn_temp_sum = torch.cat((combined_attn_temp_sum, combined_attn_temp_zero), dim=-1)
            start_head = end_head

        
        query, key, value = x.split(self.split_size, dim=2)
        if combined_attn_temp_sum is not None:
            query_lora, key_lora, value_lora = combined_attn_temp_sum.split(self.split_size, dim=2)
        query = query + query_lora
        key = key + key_lora
        value = value + value_lora
        # query: torch.Size([4, 512, 1024])
        # key: torch.Size([4, 512, 1024])
        # value: torch.Size([4, 512, 1024])

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        # _input_msk = None

        len_kv = None

        if layer_past is not None:
            # key : (batch, head, head_features, seq_length)
            # value : (batch, head, seq_length, head_features)
            # layer_past, key : (batch, head, seq_length, head_features)
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch, :, len_past, :] = key.squeeze(-1)
                past_value[_batch, :, len_past, :] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, len_kv=len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present, combined_attn_temp_sum


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.config = config

    def add_grad_hook(self, tensor_shape, device, negative_mask, start_head, end_head, n_head):
        def grad_hook(grad):
            
            recv_tensor_grad, _ = dist_recv_grad(tensor_shape, device=dist.get_rank(), src=device)

            current_device_mask = torch.zeros_like(negative_mask, dtype=torch.bool)
            current_device_mask[start_head:end_head] = negative_mask[start_head:end_head]
            restored_tensor = torch.zeros_like(grad)
            restored_tensor = restored_tensor.reshape(grad.size(0), grad.size(1), n_head, grad.size(-1) // n_head)
            restored_tensor[:, :, current_device_mask, :] = recv_tensor_grad
            restored_tensor = restored_tensor.view_as(grad)
            
            new_grad = grad + restored_tensor
            return new_grad

        return grad_hook

    def forward(self, x, layer_past=None, len_past=None, mask_vector=None):
        ln_1_feature = self.ln_1(x)
        if mask_vector is not None:
            negative_mask = ~mask_vector
            
            start_head = 0
            block_info = []
            for device_idx, device_num in enumerate(self.config.optimize_tool['head_alloc_device']):
                end_head = start_head + device_num
                tensor_to_send_device = get_current_hidden_states_by_mask(ln_1_feature, self.config, negative_mask, start_head, end_head)
                current_device_mask = negative_mask[start_head:end_head]
                mask_is_all_False = torch.all(~current_device_mask)
                if not mask_is_all_False:
                    dist_send_data(tensor_to_send_device, dst=device_idx + 1)
                    
                    if self.training:
                        ln_1_feature.retain_grad()
                        ln_1_feature.register_hook(self.add_grad_hook(tensor_to_send_device.shape, device_idx + 1, negative_mask, start_head, end_head, self.config.n_head))
                else:
                    assert tensor_to_send_device.shape[-2] == 0
                block_info.append(ln_1_feature)
                start_head = end_head
        
        
        a, present, combined_attn_temp = self.attn(ln_1_feature, layer_past=layer_past, len_past=len_past, negative_mask=negative_mask)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present, block_info, combined_attn_temp


class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        if True:  # dist.get_rank() > 0
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)
            self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config

    def forward(
            self,
            input_ids,
            position_ids=None,
            token_type_ids=None,
            past=None,
            len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length,
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1)  # .long()

        input_shape = input_ids.size()  #  position_ids: torch.Size([4, 512]) , index from 0-511
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)  #  word token embedding torch.Size([4, 512, 1024])

        position_embeds = self.wpe(position_ids)  #  word position embedding

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past=layer_past, len_past=len_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

    def send_final_hidden_states_grad_to_base_hook(self, tensor_shape, device):
        def grad_hook(grad):
            dist_send_data(grad, dst=0)

        return grad_hook

    def recv_combined_attn_grad_from_base_hook(self, tensor, device):
        def grad_hook(grad):
            recv_tensor_grad = torch.empty(tensor.shape).to(device)
            dist.recv(tensor=recv_tensor_grad, src=0)
            tensor.backward(grad)

        return grad_hook

    def send_tensor_to_lora_grad_to_base_hook(self, tensor_shape, device):
        def grad_hook(grad):
            dist_send_data(grad, dst=0, is_cal_criterion=self.config.optimize_tool['is_statistical_dict'], statistical_dict=self.config.statistical_dict, is_belong_block=True, is_backward=True)

        return grad_hook

    def send_base_first_hidden_states_grad_to_lora_hook(self, tensor_shape, device):
        def grad_hook(grad):
            dist_send_data(grad, dst=1)

        return grad_hook

    def recv_final_hidden_states_grad_from_lora_hook(self, tensor, device):
        def grad_hook(grad):
            # 接收数据
            recv_tensor_grad = torch.empty(tensor.shape).to(device)
            dist.recv(tensor=recv_tensor_grad, src=1)
            print("hello999")
            tensor.backward(grad)

        return grad_hook

    def set_statistical_dict(self, device):
        period_window = 50
        periodic_clear_epoch_key = True
        update_clear_statistical_dict(self.config.statistical_dict, 'send_data_array_per_step',
                                      'send_data_array_per_epoch', period_window,
                                      periodic_clear_epoch_key=periodic_clear_epoch_key)
        update_clear_statistical_dict(self.config.statistical_dict, 'send_grad_array_per_step',
                                      'send_grad_array_per_epoch', period_window,
                                      periodic_clear_epoch_key=periodic_clear_epoch_key)
        update_clear_statistical_dict(self.config.statistical_dict, 'recv_data_array_per_step',
                                      'recv_data_array_per_epoch', period_window,
                                      periodic_clear_epoch_key=periodic_clear_epoch_key)
        update_clear_statistical_dict(self.config.statistical_dict, 'recv_grad_array_per_step',
                                      'recv_grad_array_per_epoch', period_window,
                                      periodic_clear_epoch_key=periodic_clear_epoch_key)
        calculate_in_out_info_flag = False

        calculate_info_matrix(self.config.statistical_dict, 'send_data_array_per_step', 'send_data_array_per_epoch',
                              calculate_in_out_info_flag, device=device)
        calculate_info_matrix(self.config.statistical_dict, 'send_grad_array_per_step', 'send_grad_array_per_epoch',
                              calculate_in_out_info_flag, device=device)
        calculate_info_matrix(self.config.statistical_dict, 'recv_data_array_per_step', 'recv_data_array_per_epoch',
                              calculate_in_out_info_flag, device=device)
        calculate_info_matrix(self.config.statistical_dict, 'recv_grad_array_per_step', 'recv_grad_array_per_epoch',
                              calculate_in_out_info_flag, device=device)
        if self.config.statistical_dict['idx'] >= period_window:
            if len(self.config.statistical_dict['send_data_array_per_epoch']) % period_window == 0:
                calculate_info_matrix_time_step(self.config.statistical_dict, 'send_data_array_per_step',
                                                'send_data_array_per_epoch',
                                                calculate_in_out_info_flag, period_window, device=device)
                calculate_info_matrix_time_step(self.config.statistical_dict, 'send_grad_array_per_step',
                                                'send_grad_array_per_epoch',
                                                calculate_in_out_info_flag, period_window, device=device)
                calculate_info_matrix_time_step(self.config.statistical_dict, 'recv_data_array_per_step',
                                                'recv_data_array_per_epoch',
                                                calculate_in_out_info_flag, period_window, device=device)
                calculate_info_matrix_time_step(self.config.statistical_dict, 'recv_grad_array_per_step',
                                                'recv_grad_array_per_epoch',
                                                calculate_in_out_info_flag, period_window, device=device)

    @staticmethod
    def forward_lora(
            self,
            input_ids,
            position_ids=None,
            token_type_ids=None,
            past=None,
            len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        device = input_ids.device
        info = {}
        if dist.get_rank() == 1:
            if position_ids is None and len_past is None:
                position_ids = torch.arange(
                    past_length, input_ids.size(-1) + past_length,
                    dtype=torch.long, device=input_ids.device
                )
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            elif len_past is not None:
                position_ids = (len_past).unsqueeze(1) 

            input_shape = input_ids.size()  #  position_ids: torch.Size([4, 512]) , index from 0-511
            input_ids = input_ids.view(-1, input_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))

            inputs_embeds = self.wte(input_ids)  #  word token embedding torch.Size([4, 512, 1024])

            position_embeds = self.wpe(position_ids)  #  word position embedding

            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
                token_type_embeds = self.wte(token_type_ids)
            else:
                token_type_embeds = 0
            hidden_states = inputs_embeds + position_embeds + token_type_embeds

            hidden_states.requires_grad_(True)
            info[f'first_hidden_states_{dist.get_rank()}'] = hidden_states
            if self.config.optimize_tool['is_include_input_x']:
                self.config.optimize_tool['input_x'].append(hidden_states)
            transmit_start_time = time.time()
            dist_send_data(hidden_states, dst=0, is_cal_criterion=self.config.optimize_tool['is_statistical_dict'], statistical_dict=self.config.statistical_dict)

        presents = []
        tensor_to_lora_list = []
        combined_attn_temp_list = []
        if self.training and self.config.optimize_tool['is_statistical_dict']:
            self.set_statistical_dict(device)
        hidden_state_shape = self.config.optimize_tool['hidden_state_shape']
        layer_idx = 0
        for block, layer_past in zip(self.h, past):
            head_device_list = self.config.optimize_tool['head_alloc_device']
            head_device_idx_list = self.config.optimize_tool['head_device_idx_list']
            mask = self.config.optimize_tool['cur_step_decision_matrix'][layer_idx].to(torch.bool)
            negative_mask = ~mask

            start_head, end_head = head_device_idx_list[dist.get_rank()-1:dist.get_rank()+1]
            recv_tensor_shape = get_current_hidden_states_shape(hidden_state_shape, negative_mask, start_head, end_head)
    
            current_device_mask = negative_mask[start_head:end_head]
            mask_is_all_False = torch.all(~current_device_mask)
            if not mask_is_all_False:
                tensor_to_lora, _ = dist_recv_data(recv_tensor_shape, device, src=0, is_cal_criterion=self.config.optimize_tool['is_statistical_dict'], statistical_dict=self.config.statistical_dict, is_belong_block=True)

                tensor_to_lora.retain_grad()  # ???
                tensor_to_lora.register_hook(
                    self.send_tensor_to_lora_grad_to_base_hook(tensor_to_lora.shape, dist.get_rank()))
                
                device_num = head_device_list[dist.get_rank()-1]
                hidden_state_slipt_head_shape = hidden_state_shape[:-1] + (device_num, hidden_state_shape[-1] // self.config.n_head)
                tensor_to_lora = restore_compressed_tensor(tensor_to_lora, negative_mask, start_head, end_head, hidden_state_slipt_head_shape)
                
                tensor_to_lora_list.append(tensor_to_lora)
            else:
                assert recv_tensor_shape[-2] == 0
                tensor_to_lora_list.append(float('inf'))

            current_device_pruning_mask = mask[start_head:end_head]
            true_idx = torch.nonzero(current_device_pruning_mask).reshape(-1).tolist()
            # Pruning Rules
            pattern = re.compile('|'.join(map(str, true_idx)))
            for n, p in block.named_parameters():
                if 'lora_' in n:
                    if pattern.search(n) and len(true_idx):
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                        
            if not mask_is_all_False:
                combined_attn_temp = block.attn.c_attn(tensor_to_lora, stage=dist.get_rank(), negative_mask=negative_mask[start_head:end_head])
                
                if self.training:
                    combined_attn_temp.retain_grad()
                combined_attn_temp_list.append(combined_attn_temp)
                dist_send_data(combined_attn_temp, dst=0, is_cal_criterion=self.config.optimize_tool['is_statistical_dict'], statistical_dict=self.config.statistical_dict, is_belong_block=True)
            else:
                combined_attn_temp_list.append(float('inf'))
            layer_idx += 1

        info[f'tensor_to_lora_list_{dist.get_rank()}'] = tensor_to_lora_list
        info[f'combined_attn_temp_list_{dist.get_rank()}'] = combined_attn_temp_list

        if dist.get_rank() == 1:
            hidden_states, _ = dist_recv_data(hidden_state_shape, device, src=0, is_cal_criterion=self.config.optimize_tool['is_statistical_dict'], statistical_dict=self.config.statistical_dict)
            latency = time.time() - transmit_start_time
            if self.config.statistical_dict.get('transmit_time_per_step') is not None:
                self.config.statistical_dict['transmit_time_per_step'] = (self.config.statistical_dict[
                                                                                'transmit_time_per_step'] + latency) / 2
            else:
                self.config.statistical_dict['transmit_time_per_step'] = latency
            hidden_states.retain_grad()
            hidden_states.register_hook(
                self.send_final_hidden_states_grad_to_base_hook(hidden_states.shape, dist.get_rank()))
            info[f'final_hidden_states_{dist.get_rank()}'] = hidden_states

            hidden_states = self.ln_f(hidden_states)
            output_shape = input_shape + (hidden_states.size(-1),)
            return hidden_states.view(*output_shape), presents, info
        return None, presents, info

    @staticmethod
    def forward_base(
            self,
            input_ids,
            lm_labels,
            position_ids=None,
            token_type_ids=None,
            past=None,
            len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past.
            past_length = past[0][0].size(-2)
        device = lm_labels
        
        info = {}
        input_ids_shape = list(tuple(input_ids))
        input_ids_shape.append(self.n_embd)
        hidden_states, _ = dist_recv_data(input_ids_shape, device, src=1)
        hidden_states.retain_grad()
        hidden_states.register_hook(
            self.send_base_first_hidden_states_grad_to_lora_hook(hidden_states.shape, dist.get_rank()))
        info['base_first_hidden_states'] = hidden_states

        presents = []
        hidden_states_to_lora_list = []
        combined_attn_temp_list = []
        layer_idx = 0
        self.config.optimize_tool['vector'] = None
        for block, layer_past in zip(self.h, past):
            self.config.optimize_tool['vector'] = self.config.optimize_tool['cur_step_decision_matrix'][layer_idx].to(torch.bool)
            hidden_states, present, block_info, att_lora_info = block(hidden_states, layer_past=layer_past,
                                                                      len_past=len_past, mask_vector=self.config.optimize_tool['vector'])
            presents.append(present)
            layer_idx += 1
            hidden_states_to_lora_list.append(block_info)
            combined_attn_temp_list.append(att_lora_info)

        info['base_hidden_states_to_lora_list'] = hidden_states_to_lora_list
        info['base_combined_attn_temp_list'] = combined_attn_temp_list
        dist_send_data(hidden_states, dst=1)
        if self.training:
            hidden_states.retain_grad()
        #
        info['base_final_hidden_states'] = hidden_states

        input_shape = input_ids
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents, info


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            lora_attn_dim=0,
            lora_attn_alpha=128,
            lora_dropout=0.0,
            lora_r_dropout=0.0,
            fix_dropout=0.0,
            client_num=1,
            ref_event=None,
            optimize_tool=None,
            statistical_dict=None
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        self.lora_r_dropout = lora_r_dropout

        self.fix_dropout = fix_dropout
        self.client_num = client_num
        self.optimize_tool = optimize_tool
        self.statistical_dict = statistical_dict
        self.ref_event = ref_event


class GPT2LMModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMModel, self).__init__()
        self.transformer = GPT2Model(config)
        # if dist.get_rank() > 0:
        self.lm_head = GPT2LMHead(self.transformer.wte.weight,
                                  config)
        self.apply(self._init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(
            self,
            input_ids,
            lm_labels=None,
            lm_mask=None,
            past=None,
            len_past=None,
            label_smooth=0.0,
            is_report_accuracy=False
    ):
        if dist.get_rank() > 0:
            _batch, _len = input_ids.shape

            hidden_states, presents, lora_info = GPT2Model.forward_lora(
                self.transformer,
                input_ids, past=past, len_past=len_past)
        else:
            hidden_states, presents, base_info = GPT2Model.forward_base(
                self.transformer,
                input_ids, lm_labels, past=past, len_past=len_past)
        lm_logits = 0
        if dist.get_rank() == 1:
            lm_logits = self.lm_head(hidden_states)

            if lm_labels is not None:

                if is_report_accuracy:
                    _pred_token = torch.argmax(lm_logits, dim=-1)
                    _hit = (_pred_token == lm_labels) * lm_mask

                    _t1_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)
                    _all_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)

                    for _b in range(0, _batch):
                        for _i in range(0, _len):
                            if lm_mask[_b, _i] >= 1.0:
                                if _hit[_b, _i] > 0:
                                    _t1_acc[_b] = 1.0
                                break

                        _is_succ = True
                        for _i in range(0, _len):
                            if lm_mask[_b, _i] >= 1.0:
                                if _hit[_b, _i] <= 0:
                                    _is_succ = False
                                    break

                        if _is_succ:
                            _all_acc[_b] = 1.0

                    # _t1_acc = _t1_acc * 1.0 / _batch
                    # _all_acc = _all_acc * 1.0 / _batch

                if label_smooth > 0.0001:
                    logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                    nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -logprobs.mean(dim=-1)
                    loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                    loss = loss.view(_batch, _len)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
                    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)

                if lm_mask is None:
                    lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
                loss = loss * lm_mask

                loss = loss.sum() / (lm_mask.sum() + 0.0001)

                if is_report_accuracy:
                    return lm_logits, loss, _t1_acc, _all_acc
                else:
                    return lm_logits, loss, lora_info
        if dist.get_rank() == 0:
            return lm_logits, presents, base_info
        else:
            return lm_logits, presents, lora_info

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"

            if key.startswith("lm_head.decoder."):
                new_key = 'module.' + key

            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if key.startswith("transformer."):
                new_key = key[len("transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        self.transformer.load_state_dict(state_dict, strict=False)
        if dist.get_rank() > 0:
            self.set_tied()
