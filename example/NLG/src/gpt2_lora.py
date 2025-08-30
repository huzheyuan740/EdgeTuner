#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import numpy as np
import itertools

import torch
import random
from torch.utils.data import DataLoader, RandomSampler
import re
import h5py
from opt_pred_model_encoder import PredictNetwork

torch.set_printoptions(threshold=100000)
torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)
import torch.distributed as dist
from dist_comm_util import *

from gpu import (
    add_gpu_params,
    parse_gpu,
    distributed_opt,
    distributed_gather,
    distributed_sync,
    cleanup
)
from optimizer import (
    create_adam_optimizer,
    create_optimizer_scheduler,
    add_optimizer_params,
    create_adam_optimizer_from_args
)

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir
from datetime import datetime

from torchviz import make_dot


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

import edge_loralib as lora

parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')

parser.add_argument('--valid_data', required=True, help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'],
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')

parser.add_argument('--fp16', action='store_true', help='train model with fp16')

parser.add_argument('--log_interval', type=int, default=100, help='log interval')

parser.add_argument('--eval_interval', type=int, default=20000, help='eval interval')

parser.add_argument('--save_interval', type=int, default=500, help='save interval')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'),
                    help='working folder.')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'],
                    help='language model training objective')

parser.add_argument('--lora_dropout', default=0.0, type=float,
                    help='dropout probability for lora layers')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')

parser.add_argument('--client_num', type=int, default=1, help='number of clients.')


# influence model, calculate the influence score between two samples.
def print_args(args):
    if dist.get_rank() > 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_s(s_base, alpha, gamma_list, rho, N_pruned_lm, beta, is_init=False, s_comp=None, k_idx=None, layer_idx=None):
    """
    Eqs.(29) 
    """
    
    part1 = alpha * s_base
    gamma_m_ratio = torch.ones_like(part1)
    pruned_head_m_ratio = torch.ones_like(part1)
    begin_gamma_item = 0
    begin_N_pruned_item = 0
    if is_init:
        assert s_comp is None
        N_pruned_lm = np.zeros_like(np.array(N_pruned_lm))
        for gamma_item, N_pruned_item in zip(gamma_list, N_pruned_lm):
            gamma_m_ratio[:, :, begin_gamma_item:begin_gamma_item+gamma_item] = gamma_item / sum(gamma_list)
            pruned_head_m_ratio[:, :, begin_gamma_item:begin_gamma_item+gamma_item] = N_pruned_item / gamma_item
            begin_gamma_item += gamma_item
    else:
        assert s_comp is not None
        for gamma_item, N_pruned_item in zip(gamma_list, N_pruned_lm):
            gamma_m_ratio[k_idx, layer_idx, begin_gamma_item:begin_gamma_item+gamma_item] = gamma_item / sum(gamma_list)
            pruned_head_m_ratio[k_idx, layer_idx, begin_gamma_item:begin_gamma_item+gamma_item] = N_pruned_item / gamma_item
            begin_gamma_item += gamma_item
    
    part2 = gamma_m_ratio ** rho
    part3 = 1 + beta * pruned_head_m_ratio
    result = part1 * part2 * part3

    if s_comp is not None:
        s_comp[k_idx, layer_idx,:] = result[k_idx, layer_idx,:]
        return s_comp
    return result

def find_device_index(head_alloc_device, index):
    total = 0
    for device_index, num_heads in enumerate(head_alloc_device):
        total += num_heads
        if index < total:
            return device_index
    return None

def Pruning_baseline(s_base, config_optimize_tool, kill_rate=None):
    s_base_clone = s_base.clone()
    decision_matrix = torch.ones_like(s_base_clone)
    s_base_sum_origin = torch.sum(s_base_clone)
    head_alloc_device = config_optimize_tool['head_alloc_device']
    if kill_rate is None:
        head_reserve_device = [round(item * (config_optimize_tool['tau'])) for item in head_alloc_device]
    else:
        head_reserve_device = [item - round(item * (kill_rate.item())) for item in head_alloc_device]
    k_size, layer_size, _ = s_base_clone.shape
    for k_idx in range(k_size):
        for layer_idx in range(layer_size):
            s_base_tensor_list = s_base_clone[k_idx, layer_idx]
            start_idx = 0
            for head_alloc_item, head_reserve_item in zip(head_alloc_device, head_reserve_device):
                sliced_s_base_tensor_list = s_base_tensor_list[start_idx:start_idx+head_alloc_item]
                top_k_list, relative_top_k_idx = torch.topk(sliced_s_base_tensor_list, k=head_reserve_item)
                top_k_idx = relative_top_k_idx + start_idx
                decision_matrix[k_idx, layer_idx, top_k_idx] = 0  # keep current head
                start_idx = start_idx+head_alloc_item
    
    decision_matrix_sum = torch.sum(decision_matrix)
    decision_matrix_all = torch.sum(torch.ones_like(decision_matrix))
    s_base_clone[decision_matrix==1] = 0
    s_base_sum = torch.sum(s_base_clone)
    print(f"decision_baseline_sum:{decision_matrix_sum}, decision_baseline_all:{decision_matrix_all}， kill_rate_baseline:{decision_matrix_sum / decision_matrix_all}")
    print(f"s_base_baseline:{s_base_sum}, s_base_baseline_origin:{s_base_sum_origin}， importance_reserve_rate_baseline:{s_base_sum / s_base_sum_origin}")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    return decision_matrix

def Greedy_Pruning(s_base, s_comp, config_optimize_tool):
    # print("s_base_here:", s_base)
    s_base_clone = s_base.clone()
    decision_matrix = torch.zeros_like(s_base_clone)
    head_alloc_device = config_optimize_tool['head_alloc_device']
    s_base_sum = torch.sum(s_base_clone, dim=-1)
    s_base_sum_origin = torch.sum(s_base_clone)
    tau = config_optimize_tool['tau']
    alpha = config_optimize_tool['alpha']
    rho = config_optimize_tool['rho']
    beta = config_optimize_tool['beta']
    tau_tensor = tau * s_base_sum

    k_size, layer_size, _ = s_comp.shape
    for k_idx in range(k_size):
        for layer_idx in range(layer_size):
            N_pruned_lm = np.zeros_like(np.array(head_alloc_device))
            while s_base_sum[k_idx, layer_idx] > tau_tensor[k_idx, layer_idx]:
                layer_item_clone = s_comp[k_idx, layer_idx]
                min_s_comp = torch.min(layer_item_clone)
                min_s_comp_idx = torch.argmin(layer_item_clone)
                decision_matrix[k_idx, layer_idx, min_s_comp_idx] = 1
                device_idx = find_device_index(head_alloc_device, min_s_comp_idx)
                N_pruned_lm[device_idx] += 1
                s_comp = compute_s(
                    s_base_clone, 
                    alpha, 
                    head_alloc_device, 
                    rho, 
                    N_pruned_lm, 
                    beta,
                    s_comp=s_comp,
                    k_idx=k_idx,
                    layer_idx=layer_idx
                    )
                s_comp[decision_matrix == 1] = float('inf')
                s_base_sum[k_idx, layer_idx] -= s_base_clone[k_idx, layer_idx, min_s_comp_idx]
    decision_matrix_sum = torch.sum(decision_matrix)
    decision_matrix_all = torch.sum(torch.ones_like(decision_matrix))
    s_base_clone[decision_matrix==1] = 0
    s_base_sum = torch.sum(s_base_clone)
    kill_rate = decision_matrix_sum / decision_matrix_all
    print(f"decision_matrix_sum:{decision_matrix_sum}, decision_matrix_all:{decision_matrix_all}， kill_rate:{kill_rate}")
    print(f"s_base_sum:{s_base_sum}, s_base_sum_origin:{s_base_sum_origin}， importance_reserve_rate:{s_base_sum / s_base_sum_origin}")
    print("------------------------------------------------------")
    return decision_matrix, kill_rate


def save_data_to_hdf5(data):
    """
    save the generated data as HDF5 files
    """
    file_path = os.path.join(data['dataset_path'], 'lora_dataset_batch_4_model.h5')
    if dist.get_rank() == 1 and config.optimize_tool['is_include_input_x']:
        stacked_input_x = torch.stack(data['input_x'], dim=0)
    with h5py.File(file_path, 'a') as f:
        # init dataset
        if 'matrix' not in f:
            if config.optimize_tool['is_include_input_x']:
                input_x_dset = f.create_dataset('input_x', shape=(0,)+tuple(stacked_input_x.shape), maxshape=(None,)+tuple(stacked_input_x.shape), dtype='f4')
            base_parm_dset = f.create_dataset('base_parm', shape=(0,)+tuple(data['base_parm'].shape), maxshape=(None,)+tuple(data['base_parm'].shape), dtype='f4')
            lora_parm_dset = f.create_dataset('lora_parm', shape=(0,)+tuple(data['lora_parm'].shape), maxshape=(None,)+tuple(data['lora_parm'].shape), dtype='f4')
            lora_grad_dset = f.create_dataset('lora_grad', shape=(0,)+tuple(data['lora_grad'].shape), maxshape=(None,)+tuple(data['lora_grad'].shape), dtype='f4')
            matrix_dset = f.create_dataset('matrix', shape=(0,)+tuple(data['matrix'].shape), maxshape=(None,)+tuple(data['matrix'].shape), dtype='f4')
            epoch_dset = f.create_dataset('epoch', shape=(0,), maxshape=(None,), dtype='i4')
        else:
            if config.optimize_tool['is_include_input_x']:
                input_x_dset = f['input_x']
            base_parm_dset = f['base_parm']
            lora_parm_dset = f['lora_parm']
            lora_grad_dset = f['lora_grad']
            matrix_dset = f['matrix']
            epoch_dset = f['epoch']

        # add dataset
        index = len(matrix_dset)
        if config.optimize_tool['is_include_input_x']:
            input_x_dset.resize(index + 1, axis=0)
        base_parm_dset.resize(index + 1, axis=0)
        lora_parm_dset.resize(index + 1, axis=0)
        lora_grad_dset.resize(index + 1, axis=0)
        matrix_dset.resize(index + 1, axis=0)
        epoch_dset.resize(index + 1, axis=0)

        # save dataset
        if config.optimize_tool['is_include_input_x']:
            input_x_dset[index] = stacked_input_x.cpu().detach().numpy()
        base_parm_dset[index] = data['base_parm'].detach().cpu().numpy()
        lora_parm_dset[index] = data['lora_parm'].detach().cpu().numpy()
        lora_grad_dset[index] = data['lora_grad'].detach().cpu().numpy()
        matrix_dset[index] = data['matrix'].detach().cpu().numpy()
        epoch_dset[index] = data['epoch']


def post_cal_Frobenius_matrix(matrix):
    matrix = torch.sqrt(matrix) / 3
    min_val = torch.min(matrix)
    max_val = torch.max(matrix)
    if max_val == min_val:
        normalized_tensor = torch.zeros_like(matrix)
    else:
        normalized_tensor = (matrix - min_val) / (max_val - min_val)
    Frobenius_matrix_normalized = normalized_tensor
    return Frobenius_matrix_normalized


def update_model_by_optimizer(_optimizer, _model, _schedule, args, is_update=True):
    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()
        _optimizer.zero_grad()


def update_model_by_optimizer_collect_data(_optimizer, _model, _schedule, args, is_update=True):
    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()

        if config.optimize_tool['is_pred_net_data_collect'] and dist.get_rank() > 0:  # default rank1
            k_steps_idx = config.statistical_dict['idx']
            # get Head Importance
            n_layer = config.n_layer
            n_head = config.n_head
            Frobenius_matrix = torch.zeros((n_layer, n_head), dtype=torch.float, device=args.device)
            Frobenius_matrix_base_parm = torch.zeros((n_layer, n_head), dtype=torch.float, device=args.device)
            Frobenius_matrix_lora_parm = torch.zeros((n_layer, n_head), dtype=torch.float, device=args.device)
            head_device_idx_list = config.optimize_tool['head_device_idx_list']
            start_head, end_head = head_device_idx_list[dist.get_rank()-1:dist.get_rank()+1]
            for n, p in _model.named_parameters():
                if 'lora_' in n and p.grad is not None:
                    numbers = re.findall(r'\d+', n)
                    layer_idx = int(numbers[0])
                    head_idx = int(numbers[1])
                    abs_head_idx = start_head + head_idx
                    Frobenius_matrix[layer_idx, abs_head_idx] += torch.sum(p.grad ** 2)  # Lora A + Lora B
                    if k_steps_idx % config.optimize_tool['history_k_steps'] == (config.optimize_tool['history_k_steps'] - 1):
                        Frobenius_matrix_lora_parm[layer_idx, abs_head_idx] += torch.sum(p ** 2)  # Lora A + Lora B
                elif 'lora_' not in n and 'c_attn.weight' in n:
                    numbers = re.findall(r'\d+', n)
                    layer_idx = int(numbers[0])
                    if k_steps_idx == 0:
                        Frobenius_matrix_base_parm[layer_idx, :] = torch.sum(p ** 2)
            Frobenius_matrix_normalized = post_cal_Frobenius_matrix(Frobenius_matrix)
            if k_steps_idx == 0:
                Frobenius_matrix_base_parm_normalized = post_cal_Frobenius_matrix(Frobenius_matrix_base_parm)
                config.optimize_tool['base_parm'] = Frobenius_matrix_base_parm_normalized
            
            config.optimize_tool['matrix'][k_steps_idx%config.optimize_tool['history_k_steps'],:,:] = Frobenius_matrix_normalized
            config.optimize_tool['lora_grad'][k_steps_idx%config.optimize_tool['history_k_steps'],:,:] = Frobenius_matrix_normalized

            _optimizer.zero_grad()

            if _schedule is not None:
                _schedule.step()

            if k_steps_idx % config.optimize_tool['history_k_steps'] == (config.optimize_tool['history_k_steps'] - 1):
                Frobenius_matrix_lora_parm_normalized = post_cal_Frobenius_matrix(Frobenius_matrix_lora_parm)
                config.optimize_tool['lora_parm'] = Frobenius_matrix_lora_parm_normalized
                
                k_steps_matrix = config.optimize_tool['matrix']
                lambda_k = config.optimize_tool['lambda']
                for each_k_idx in range(k_steps_matrix.shape[0]):
                    if each_k_idx > 0:
                        k_steps_matrix[each_k_idx] = lambda_k * k_steps_matrix[each_k_idx] + (1 - lambda_k) *k_steps_matrix[each_k_idx-1]
                config.optimize_tool['matrix'] = k_steps_matrix

                if dist.get_rank() > 0 and dist.get_rank() != 1:
                    dist.send(config.optimize_tool['lora_parm'], dst=1)
                    dist.send(config.optimize_tool['lora_grad'], dst=1)
                    dist.send(config.optimize_tool['matrix'], dst=1)

                
                if dist.get_rank() == 1:
                    for rand_idx in [2, 3]:
                        lora_parm = torch.empty_like(config.optimize_tool['lora_parm'])
                        dist.recv(lora_parm, src=rand_idx)
                        config.optimize_tool['lora_parm'] += lora_parm

                        lora_grad = torch.empty_like(config.optimize_tool['lora_grad'])
                        dist.recv(lora_grad, src=rand_idx)
                        config.optimize_tool['lora_grad'] += lora_grad
                        
                        matrix = torch.empty_like(config.optimize_tool['matrix'])
                        dist.recv(matrix, src=rand_idx)
                        config.optimize_tool['matrix'] += matrix
                        
                    save_data_to_hdf5(config.optimize_tool)
                config.optimize_tool['input_x'].clear()
                return config.optimize_tool['matrix']

        _optimizer.zero_grad()


def update_model_by_optimizer_collect_data_parallel(_optimizer, _model, _schedule, args, is_update=True):
    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()

        if config.optimize_tool['is_pred_net_data_collect'] and dist.get_rank() > 0:
            k_steps_idx = config.statistical_dict['idx']
            # get Head Importance
            n_layer = config.n_layer
            n_head = config.n_head
            each_head_n_embd = config.n_embd // config.n_head
            Frobenius_matrix = torch.zeros((n_layer, n_head), dtype=torch.float, device=args.device)
            Frobenius_matrix_base_parm = torch.zeros((n_layer, n_head), dtype=torch.float, device=args.device)
            Frobenius_matrix_lora_parm = torch.zeros((n_layer, n_head), dtype=torch.float, device=args.device)
            head_device_idx_list = config.optimize_tool['head_device_idx_list']
            start_head, end_head = head_device_idx_list[dist.get_rank()-1:dist.get_rank()+1]
            for n, p in _model.named_parameters():
                for idx, abs_head_idx in enumerate(range(start_head, end_head)):
                    if 'lora_' in n and p.grad is not None:
                        # print(f"==rank:{dist.get_rank()}, name:{n}, parm:{p.size()}, grad?:{p.requires_grad}")
                        numbers = re.findall(r'\d+', n)
                        layer_idx = int(numbers[0])
                        # head_idx = int(numbers[1])
                        idx_chunk = idx * each_head_n_embd
                        if 'lora_A' in n:
                            p_split_head = p[:, idx_chunk:idx_chunk+each_head_n_embd]
                            p_split_head_grad = p.grad[:, idx_chunk:idx_chunk+each_head_n_embd]
                        elif 'lora_B' in n:
                            output_shape = p.size()[0] // 2
                            p_split_head1 = p[idx_chunk:idx_chunk+each_head_n_embd, :]
                            p_split_head2 = p[output_shape+idx_chunk:output_shape+idx_chunk+each_head_n_embd, :]
                            p_split_head = torch.cat([p_split_head1, p_split_head2], dim=0)

                            p_split_head1_grad = p.grad[idx_chunk:idx_chunk+each_head_n_embd, :]
                            p_split_head2_grad = p.grad[output_shape+idx_chunk:output_shape+idx_chunk+each_head_n_embd, :]
                            p_split_head_grad = torch.cat([p_split_head1_grad, p_split_head2_grad], dim=0)
                        Frobenius_matrix[layer_idx, abs_head_idx] += torch.sum(p_split_head_grad ** 2)  # Lora A + Lora B
                        if k_steps_idx % config.optimize_tool['history_k_steps'] == (config.optimize_tool['history_k_steps'] - 1):
                            Frobenius_matrix_lora_parm[layer_idx, abs_head_idx] += torch.sum(p_split_head ** 2)  # Lora A + Lora B
                    elif 'lora_' not in n and 'c_attn.weight' in n:
                        if k_steps_idx == 0:
                            numbers = re.findall(r'\d+', n)
                            layer_idx = int(numbers[0])
                            Frobenius_matrix_base_parm[layer_idx, :] = torch.sum(p ** 2)
            Frobenius_matrix_normalized = post_cal_Frobenius_matrix(Frobenius_matrix)
            if k_steps_idx == 0:
                Frobenius_matrix_base_parm_normalized = post_cal_Frobenius_matrix(Frobenius_matrix_base_parm)
                config.optimize_tool['base_parm'] = Frobenius_matrix_base_parm_normalized
            
            config.optimize_tool['matrix'][k_steps_idx%config.optimize_tool['history_k_steps'],:,:] = Frobenius_matrix_normalized
            config.optimize_tool['lora_grad'][k_steps_idx%config.optimize_tool['history_k_steps'],:,:] = Frobenius_matrix_normalized

            _optimizer.zero_grad()

            if _schedule is not None:
                _schedule.step()

            if k_steps_idx % config.optimize_tool['history_k_steps'] == (config.optimize_tool['history_k_steps'] - 1):
                Frobenius_matrix_lora_parm_normalized = post_cal_Frobenius_matrix(Frobenius_matrix_lora_parm)
                config.optimize_tool['lora_parm'] = Frobenius_matrix_lora_parm_normalized
                
                k_steps_matrix = config.optimize_tool['matrix']
                lambda_k = config.optimize_tool['lambda']
                for each_k_idx in range(k_steps_matrix.shape[0]):
                    if each_k_idx > 0:
                        k_steps_matrix[each_k_idx] = lambda_k * k_steps_matrix[each_k_idx] + (1 - lambda_k) *k_steps_matrix[each_k_idx-1]
                config.optimize_tool['matrix'] = k_steps_matrix

                if dist.get_rank() > 0 and dist.get_rank() != 1:
                    dist.send(config.optimize_tool['lora_parm'], dst=1)
                    dist.send(config.optimize_tool['lora_grad'], dst=1)
                    dist.send(config.optimize_tool['matrix'], dst=1)

                
                if dist.get_rank() == 1:
                    for rand_idx in [2, 3]:
                        lora_parm = torch.empty_like(config.optimize_tool['lora_parm'])
                        dist.recv(lora_parm, src=rand_idx)
                        config.optimize_tool['lora_parm'] += lora_parm

                        lora_grad = torch.empty_like(config.optimize_tool['lora_grad'])
                        dist.recv(lora_grad, src=rand_idx)
                        config.optimize_tool['lora_grad'] += lora_grad
                        
                        matrix = torch.empty_like(config.optimize_tool['matrix'])
                        dist.recv(matrix, src=rand_idx)
                        config.optimize_tool['matrix'] += matrix
                    save_data_to_hdf5(config.optimize_tool)
                config.optimize_tool['input_x'].clear()
                return config.optimize_tool['matrix']

        _optimizer.zero_grad()

def optimizer_step(_loss, _optimizer, _model, _schedule, args, lora_info, is_update=True):
    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else:
        if dist.get_rank() == 1: 
            _loss.backward()
            first_hidden_states = lora_info[f'first_hidden_states_{dist.get_rank()}']

        combined_attn_temp_list = lora_info[f'combined_attn_temp_list_{dist.get_rank()}']

        for block_idx in reversed(range(len(combined_attn_temp_list))):
            mask = config.optimize_tool['cur_step_decision_matrix'][block_idx].to(torch.bool)
            negative_mask = ~mask
            head_device_idx_list = config.optimize_tool['head_device_idx_list']
            start_head, end_head = head_device_idx_list[dist.get_rank()-1:dist.get_rank()+1]
            current_device_mask = negative_mask[start_head:end_head]
            mask_is_all_False = torch.all(~current_device_mask)

            if not mask_is_all_False:
                combined_attn_temp_grad, _ = dist_recv_grad(combined_attn_temp_list[block_idx].shape,
                                                            torch.device('cuda', int(os.environ['LOCAL_RANK'])), src=0,
                                                            is_cal_criterion=config.optimize_tool['is_statistical_dict'], statistical_dict=config.statistical_dict,
                                                            is_belong_block=True, is_backward=True)
                combined_attn_temp_list[block_idx].backward(combined_attn_temp_grad)
        
        if dist.get_rank() == 1:
            first_hidden_states_grad, _ = dist_recv_grad(first_hidden_states.shape, torch.device('cuda', int(os.environ['LOCAL_RANK'])),
                                                        src=0, is_cal_criterion=config.optimize_tool['is_statistical_dict'], statistical_dict=config.statistical_dict)
            
            first_hidden_states.backward(first_hidden_states_grad)
        if config.optimize_tool['is_pred_net_data_collect']:
            return update_model_by_optimizer_collect_data_parallel(_optimizer, _model, _schedule, args, is_update)
        else:
            update_model_by_optimizer(_optimizer, _model, _schedule, args, is_update)

    if _schedule is not None:
        _schedule.step()

    return None


def clear_all_grad(base_info):
    final_hidden_states = base_info['base_final_hidden_states']
    combined_attn_temp_list = base_info['base_combined_attn_temp_list']
    hidden_states_to_lora_list = base_info['base_hidden_states_to_lora_list']
    first_hidden_states = base_info['base_first_hidden_states']
    final_hidden_states.grad.zero_()
    first_hidden_states.grad.zero_()
    for block_idx in reversed(range(len(combined_attn_temp_list))):
        combined_attn_temp_list[block_idx].grad.zero_()
        hidden_states_to_lora_list[block_idx].grad.zero_()


def base_optimizer_step(_optimizer, _model, _schedule, args, base_info, is_update=True):
    final_hidden_states = base_info['base_final_hidden_states']
    final_hidden_states_grad, _ = dist_recv_grad(final_hidden_states.shape, torch.device('cuda', int(os.environ['LOCAL_RANK'])),
                                                 src=1)

    final_hidden_states.backward(final_hidden_states_grad)

def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            if dist.get_rank() > 0:
                _lm_logits, _loss, lora_info = model(
                    _input, lm_labels=_target, lm_mask=_msk,
                )
            else:
                lm_logits, presents, base_info = model(
                    data['input'].size(), lm_labels=args.device, lm_mask=None,
                )
            if dist.get_rank() == 1:
                loss = _loss.mean()

                avg_lm_loss.update(loss.item())

                if idx % 100 == 0:
                    print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)

def load_model(model_path, k, m_size, seq_length, embedding_dim, layer_num, head_num):
    model = PredictNetwork(k, m_size, seq_length, embedding_dim, layer_num, head_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss


def train_validate(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        args,
        train_step=0,
        epoch=0
):
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    for idx, data in enumerate(train_loader):
        if dist.get_rank() > 0:
            config.statistical_dict['idx'] = idx
            # print("idx:", idx)
        data = {key: value for key, value in data.items()}
        # print(f"-----------------------------xxx-rank-{dist.get_rank()}, train_step:{train_step}")

        if dist.get_rank() > 0:
            if config.statistical_dict.get('transmit_datasize_logic') is not None:
                if config.statistical_dict.get('transmit_datasize_logic_per_epoch') is not None:
                    config.statistical_dict['transmit_datasize_logic_per_epoch'] = (
                            config.statistical_dict['transmit_datasize_logic_per_epoch'] +
                            config.statistical_dict['transmit_datasize_logic'])
                else:
                    config.statistical_dict['transmit_datasize_logic_per_epoch'] = config.statistical_dict[
                        'transmit_datasize_logic']
                config.statistical_dict['transmit_datasize_logic'] = None

            _input = data['input'].to(
                args.device)  
            _target = data['target'].to(args.device)  
            _msk = data['mask'].to(
                args.device)

        if dist.get_rank() > 0:
            _lm_logits, _lm_loss, lora_info = model(
                _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
            )
            if dist.get_rank() == 1:
                _lm_loss = _lm_loss.mean()
        else:
            lm_logits, presents, base_info = model(
                data['input'].size(), lm_labels=args.device, lm_mask=None, label_smooth=args.label_smooth
            )

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        if dist.get_rank() > 0:
            if dist.get_rank() == 1:
                avg_lm_loss.update(_lm_loss.item())
            else:
                _lm_loss = 0
            k_steps_matrix = optimizer_step(
                _lm_loss / (args.grad_acc), optimizer, model, scheduler, args, lora_info, is_update=is_update,
            )
            
            if k_steps_matrix is not None and config.optimize_tool['is_cal_s_comp']:
                assert sum(config.optimize_tool['head_alloc_device']) == config.n_head
                k_steps_matrix_comp_init = compute_s(
                    k_steps_matrix, 
                    config.optimize_tool['alpha'], 
                    config.optimize_tool['head_alloc_device'], 
                    config.optimize_tool['rho'], 
                    config.optimize_tool['head_pruned_device'], 
                    config.optimize_tool['beta'],
                    is_init=True
                    )
                config.optimize_tool['decision_matrix'], kill_rate = Greedy_Pruning(k_steps_matrix, k_steps_matrix_comp_init, config.optimize_tool)
        elif dist.get_rank() == 0:
            base_optimizer_step(
                optimizer, model, scheduler, args, base_info,
                is_update=is_update,
            )

        if train_step % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | {idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

            if dist.get_rank() == 1:
                print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()

        if train_step % args.save_interval == 0:
            if dist.get_rank() > 0:
                model_path = os.path.join(args.work_dir, f'model.{train_step}_lora_{dist.get_rank()}.pt')
                print('saving checkpoint', model_path)
                if dist.get_rank() == 1:
                    torch.save({'model_state_dict': model.state_dict()}, model_path)
                else:
                    torch.save({'model_state_dict': lora.lora_state_dict(model)}, model_path)
            distributed_sync(args)

        # evaluation interval
        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()

            print("begin_eval==================")
            valid_loss, valid_ppl = evaluate(model, valid_loader, args)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl

            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '

            if dist.get_rank() > 0:
                print('-' * 100)
                print(log_str)
                print('-' * 100)

            model.train()
            distributed_sync(args)

        if train_step == args.max_step:
            break

    if dist.get_rank() > 0:
        model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path)
    distributed_sync(args)
    return train_step


if __name__ == '__main__':
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)
    # ref_event = torch.cuda.Event(enable_timing=True)
    # ref_event.record()
    ref_event = time.time()

    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if dist.get_rank() == 0:
        args.logging = create_exp_dir(args.work_dir)

    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len,
        joint_lm=args.obj == 'jlm'
    )

    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0,
        shuffle=False, pin_memory=False, drop_last=True,
    )

    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0,
        shuffle=False, pin_memory=False, drop_last=False,
    )

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            client_num=args.client_num,
            ref_event=ref_event,
            optimize_tool = {},
            statistical_dict={}
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            client_num=args.client_num,
            ref_event=ref_event,
            optimize_tool = {},
            statistical_dict={}
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            client_num=args.client_num,
            ref_event=ref_event,
            optimize_tool = {},
            statistical_dict={}
        )

    cur_optimize_matrix = torch.randint(0, 2, size=(config.n_layer, config.n_head))
    cur_optimize_matrix = torch.zeros_like(cur_optimize_matrix)
    config.optimize_tool['is_statistical_dict'] = False
    config.optimize_tool['is_pred_net_data_collect'] = False
    config.optimize_tool['is_include_input_x'] = False 
    config.optimize_tool['is_cal_s_comp'] = False
    config.optimize_tool['cur_step_decision_matrix'] = cur_optimize_matrix
    
    config.optimize_tool['vector'] = None
    config.optimize_tool['head_alloc_device'] = [8,5,3]  # head alloction for each device in Eqs.(3)
    head_device_idx_list = [0] * (len(config.optimize_tool['head_alloc_device'])+1)
    start_head = 0
    for device_idx, device_num in enumerate(config.optimize_tool['head_alloc_device']):
        end_head = start_head + device_num
        start_head = end_head
        head_device_idx_list[device_idx+1] = start_head
    config.optimize_tool['head_device_idx_list'] = head_device_idx_list
    

    config.optimize_tool['hidden_state_shape'] = torch.Size((args.train_batch_size, args.seq_len, config.n_embd))
    config.optimize_tool['history_k_steps'] = 10
    config.optimize_tool['lambda'] = 0.8 
    config.optimize_tool['alpha'] = 1
    config.optimize_tool['rho'] = 0.2
    config.optimize_tool['beta'] = 0.3
    config.optimize_tool['tau'] = 0.7  # keep X% head importance
    config.optimize_tool['eta'] = 0.4  # Layer-Wise Pruning
    config.optimize_tool['head_pruned_device'] = config.optimize_tool['head_alloc_device'] * 0# [round(item * config.optimize_tool['beta']) for item in config.optimize_tool['head_alloc_device']]
    optimize_matrix = torch.zeros((config.optimize_tool['history_k_steps'], config.n_layer, config.n_head), dtype=torch.float, device=args.device)
    config.optimize_tool['matrix'] = optimize_matrix 
    config.optimize_tool['decision_matrix'] = torch.zeros_like(optimize_matrix)

    # part of input of iHeadPruner
    config.optimize_tool['input_x'] = []
    config.optimize_tool['base_parm'] = None
    config.optimize_tool['lora_parm'] = None
    config.optimize_tool['lora_grad'] = torch.zeros_like(optimize_matrix)
    config.optimize_tool['epoch'] = None
    config.optimize_tool['dataset_path'] = './custom_data'
    
    # Load iHeadPruner model
    pruner_model, _ = load_model(config.optimize_tool['dataset_path'], 
                                                   config.optimize_tool['history_k_steps'], 
                                                   args.train_batch_size, args.seq_len, config.n_embd, 
                                                   config.n_layer, config.n_head)
    

    vis_save_path = None
    if dist.get_rank() > 0:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        vis_save_path = os.path.join(args.work_dir, f'{current_time}')
    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print(f'loading model pretrained weight.rank:{args.dist.get_rank()}')
        lm_net.load_weight(torch.load(args.init_checkpoint))

    lm_net = lm_net.to(args.local_rank)
    
    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)
    optimizer = create_adam_optimizer_from_args(lm_net, args)

    if args.max_step is None:
        local_world_size = 1
        args.max_step = (args.max_epoch * train_data.num_batches + local_world_size - 1) // local_world_size

    scheduler = create_optimizer_scheduler(optimizer, args)
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            if dist.get_rank() > 0:
                config.statistical_dict.clear()
                config.statistical_dict['vis_save_path'] = vis_save_path
                config.statistical_dict['epoch'] = epoch
                config.optimize_tool['epoch'] = epoch
            train_step = train_validate(
                lm_net, optimizer, scheduler, train_loader, valid_loader, args,
                train_step=train_step, epoch=epoch
            )
            if dist.get_rank() > 0:
                if config.optimize_tool['is_statistical_dict']:
                    save_vis_file(config.statistical_dict)
                    save_vis_pic(config.statistical_dict, step_idx=config.statistical_dict['idx'])

            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                if dist.get_rank() > 0:
                    print('-' * 100)
                    print('End of training')
                break
    except KeyboardInterrupt:
        if dist.get_rank() > 0:
            print('-' * 100)
            print('Exiting from training early')

    distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)
