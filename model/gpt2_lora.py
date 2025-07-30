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

# 获取当前脚本的目录，然后向上移动两级到edge_lore目录
# print("sys.path:", sys.path)
current_dir = os.path.dirname(os.path.abspath(__file__))
# print("current_dir:", current_dir)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# print("project_root:", project_root)
sys.path.append(project_root)
# print("sys.path.new:", sys.path)

# 现在你可以导入edge_loralib中的模块了
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

parser.add_argument('--log_interval', type=int, default=5, help='log interval')

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
    实现给定公式的计算
    :param lh_base: s_base^{l,h}，公式中的基础值，torch.Tensor类型
    :param alpha: 公式中的alpha系数
    :param gamma_list: 公式中 \sum_{j \in \mathcal{M}} \gamma_{ij} 里的 \gamma_{ij} 列表，torch.Tensor类型
    :param rho: 公式中的rho指数
    :param N_pruned_lm: N_pruned^{l,m}，公式中的剪枝数量，torch.Tensor类型
    :param beta: 公式中的beta系数
    :param H_m: H_m，公式中的分母值
    :return: 按照公式计算得到的结果，torch.Tensor类型
    """
    
    part1 = alpha * s_base
    gamma_m_ratio = torch.ones_like(part1)
    pruned_head_m_ratio = torch.ones_like(part1)
    begin_gamma_item = 0
    begin_N_pruned_item = 0
    if is_init:
        assert s_comp is None
        N_pruned_lm = np.zeros_like(np.array(N_pruned_lm))
        # print("N_pruned_lm:", N_pruned_lm)
        for gamma_item, N_pruned_item in zip(gamma_list, N_pruned_lm):
            gamma_m_ratio[:, :, begin_gamma_item:begin_gamma_item+gamma_item] = gamma_item / sum(gamma_list)
            pruned_head_m_ratio[:, :, begin_gamma_item:begin_gamma_item+gamma_item] = N_pruned_item / gamma_item
            # print(f"N_pruned_item:{N_pruned_item}, gamma_item:{gamma_item}, ratio:{N_pruned_item / gamma_item}")
            begin_gamma_item += gamma_item
    else:
        assert s_comp is not None
        # print("N_pruned_lm_each:", N_pruned_lm)
        for gamma_item, N_pruned_item in zip(gamma_list, N_pruned_lm):
            gamma_m_ratio[k_idx, layer_idx, begin_gamma_item:begin_gamma_item+gamma_item] = gamma_item / sum(gamma_list)
            pruned_head_m_ratio[k_idx, layer_idx, begin_gamma_item:begin_gamma_item+gamma_item] = N_pruned_item / gamma_item
            # print(f"N_pruned_item:{N_pruned_item}, gamma_item:{gamma_item}, ratio:{N_pruned_item / gamma_item}")
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
    # print("config_optimize_tool['tau']:", config_optimize_tool['tau'])
    print("kill_rate:", kill_rate)
    if kill_rate is None:
        head_reserve_device = [round(item * (config_optimize_tool['tau'])) for item in head_alloc_device]
    else:
        head_reserve_device = [item - round(item * (kill_rate.item())) for item in head_alloc_device]
    k_size, layer_size, _ = s_base_clone.shape
    for k_idx in range(k_size):
        for layer_idx in range(layer_size):
            s_base_tensor_list = s_base_clone[k_idx, layer_idx]
            # print("s_base_tensor_list:", s_base_tensor_list)
            # print("head_reserve_device:", head_reserve_device)
            start_idx = 0
            for head_alloc_item, head_reserve_item in zip(head_alloc_device, head_reserve_device):
                sliced_s_base_tensor_list = s_base_tensor_list[start_idx:start_idx+head_alloc_item]
                top_k_list, relative_top_k_idx = torch.topk(sliced_s_base_tensor_list, k=head_reserve_item)
                top_k_idx = relative_top_k_idx + start_idx
                decision_matrix[k_idx, layer_idx, top_k_idx] = 0  # 等于0表示保留当前head
                start_idx = start_idx+head_alloc_item
            # print("decision_matrix:", decision_matrix[k_idx, layer_idx, :])
    
    # print("decision_baseline:\n", decision_matrix)
    decision_matrix_sum = torch.sum(decision_matrix)
    decision_matrix_all = torch.sum(torch.ones_like(decision_matrix))
    s_base_clone[decision_matrix==1] = 0  # 把哪些等于1需要干掉的head对应的值设为0
    # print("s_base_clone:", s_base_clone)
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
    # print("s_base:\n", s_base)
    # print("s_base_sum:\n", s_base_sum)
    # print("s_comp:\n", s_comp)
    tau = config_optimize_tool['tau']
    alpha = config_optimize_tool['alpha']
    rho = config_optimize_tool['rho']
    beta = config_optimize_tool['beta']
    tau_tensor = tau * s_base_sum

    k_size, layer_size, _ = s_comp.shape
    for k_idx in range(k_size):
        for layer_idx in range(layer_size):
            # print("================layer_idx:==============", layer_idx)
            N_pruned_lm = np.zeros_like(np.array(head_alloc_device))
            while s_base_sum[k_idx, layer_idx] > tau_tensor[k_idx, layer_idx]:
                layer_item_clone = s_comp[k_idx, layer_idx]
                min_s_comp = torch.min(layer_item_clone)
                min_s_comp_idx = torch.argmin(layer_item_clone)
                # print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                # print("layer_item_clone:", layer_item_clone)
                # print("min_s_comp:", min_s_comp)
                # print("min_s_comp_idx:", min_s_comp_idx)
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
                
                # TODO 列举一下tau和哪些因素有关，显示预测梯度了，那训练集到底是以初始为准还是以训练稳定为准？
                s_base_sum[k_idx, layer_idx] -= s_base_clone[k_idx, layer_idx, min_s_comp_idx]
                # print("s_base_sum:", s_base_sum)
    # print("s_comp:\n", s_comp)
    # print("decision_matrix:\n", decision_matrix)
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
    将生成的数据保存到HDF5文件中
    :param data: 生成的数据字典
    :param file_path: HDF5文件路径
    """
    file_path = os.path.join(data['dataset_path'], 'lora_dataset_batch_4_model_618_test.h5')
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    if dist.get_rank() == 1 and config.optimize_tool['is_include_input_x']:
        stacked_input_x = torch.stack(data['input_x'], dim=0)
    # print("stacked_input_x:", stacked_input_x.shape)
    # print("data['base_parm']:", data['base_parm'].shape)
    # print("data['lora_parm']:", data['lora_parm'].shape)
    # print("data['lora_grad']:", data['lora_grad'].shape)
    # print("data['matrix']:", data['matrix'].shape)
    with h5py.File(file_path, 'a') as f:
        # 初始化数据集（如果不存在）
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

        # 扩展数据集
        index = len(matrix_dset)
        if config.optimize_tool['is_include_input_x']:
            input_x_dset.resize(index + 1, axis=0)
        base_parm_dset.resize(index + 1, axis=0)
        lora_parm_dset.resize(index + 1, axis=0)
        lora_grad_dset.resize(index + 1, axis=0)
        matrix_dset.resize(index + 1, axis=0)
        epoch_dset.resize(index + 1, axis=0)

        # 保存数据
        if config.optimize_tool['is_include_input_x']:
            input_x_dset[index] = stacked_input_x.cpu().detach().numpy()
        base_parm_dset[index] = data['base_parm'].detach().cpu().numpy()
        lora_parm_dset[index] = data['lora_parm'].detach().cpu().numpy()
        lora_grad_dset[index] = data['lora_grad'].detach().cpu().numpy()
        matrix_dset[index] = data['matrix'].detach().cpu().numpy()
        epoch_dset[index] = data['epoch']
        print("matrix_dset:", matrix_dset.shape)


def post_cal_Frobenius_matrix(matrix):
    matrix = torch.sqrt(matrix) / 3
    # print("Frobenius_matrix:", Frobenius_matrix)
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
        _optimizer.zero_grad()  # TODO Order maters? or need it?


def update_model_by_optimizer_collect_data(_optimizer, _model, _schedule, args, is_update=True):
    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()

        if config.optimize_tool['is_pred_net_data_collect'] and dist.get_rank() > 0:  # 需要根据实际指定 目前默认为rank为1的设备
            k_steps_idx = config.statistical_dict['idx']
            # 开始统计初始训练重要度
            # model_config = _model.module.transformer.config  # TODO 分布式情况_model.module可能要修改
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
                    # print(f"rank:{dist.get_rank()}, name:{n}, parm:{p.size()}, grad?:{p.requires_grad}")  # p.grad.shape
                    Frobenius_matrix[layer_idx, abs_head_idx] += torch.sum(p.grad ** 2)  # Lora A + Lora B
                    if k_steps_idx % config.optimize_tool['history_k_steps'] == (config.optimize_tool['history_k_steps'] - 1):
                        Frobenius_matrix_lora_parm[layer_idx, abs_head_idx] += torch.sum(p ** 2)  # Lora A + Lora B
                elif 'lora_' not in n and 'c_attn.weight' in n:
                    # print(f"name:{n}, parm:{p.size()}")
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
                # print("self.config.optimize_tool['input_x']:", config.optimize_tool['input_x'][0].shape)
                
                k_steps_matrix = config.optimize_tool['matrix']
                # min_val = torch.min(k_steps_matrix)
                # max_val = torch.max(k_steps_matrix)
                # if max_val == min_val:
                #     normalized_tensor = torch.zeros_like(k_steps_matrix)
                # else:
                #     normalized_tensor = (k_steps_matrix - min_val) / (max_val - min_val)
                # config.optimize_tool['matrix'] = normalized_tensor
                # print("k_steps_idx:", k_steps_idx)
                # print("config.optimize_tool['matrix']:", config.optimize_tool['matrix'])
                lambda_k = config.optimize_tool['lambda']
                for each_k_idx in range(k_steps_matrix.shape[0]):
                    if each_k_idx > 0:
                        k_steps_matrix[each_k_idx] = lambda_k * k_steps_matrix[each_k_idx] + (1 - lambda_k) *k_steps_matrix[each_k_idx-1]
                config.optimize_tool['matrix'] = k_steps_matrix

                if dist.get_rank() > 0 and dist.get_rank() != 1:
                    dist.send(config.optimize_tool['lora_parm'], dst=1)
                    dist.send(config.optimize_tool['lora_grad'], dst=1)
                    dist.send(config.optimize_tool['matrix'], dst=1)

                
                if dist.get_rank() == 1:   # 需要根据实际指定 目前默认为rank为1的设备
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
                        # print("recv:", rand_idx)
                        
                    # print(f"rank:{dist.get_rank()}, matrix\n:{config.optimize_tool['matrix'][-1]}")
                    save_data_to_hdf5(config.optimize_tool)
                config.optimize_tool['input_x'].clear()
                return config.optimize_tool['matrix']

        _optimizer.zero_grad()  # TODO Order maters? or need it?


def update_model_by_optimizer_collect_data_parallel(_optimizer, _model, _schedule, args, is_update=True):
    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()

        if config.optimize_tool['is_pred_net_data_collect'] and dist.get_rank() > 0:  # 需要根据实际指定 目前默认为rank为1的设备
            k_steps_idx = config.statistical_dict['idx']
            # 开始统计初始训练重要度
            # model_config = _model.module.transformer.config  # TODO 分布式情况_model.module可能要修改 
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
                        # print(f"name:{n}, parm:{p.size()}")
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
                # print("self.config.optimize_tool['input_x']:", config.optimize_tool['input_x'][0].shape)
                
                k_steps_matrix = config.optimize_tool['matrix']
                # min_val = torch.min(k_steps_matrix)
                # max_val = torch.max(k_steps_matrix)
                # if max_val == min_val:
                #     normalized_tensor = torch.zeros_like(k_steps_matrix)
                # else:
                #     normalized_tensor = (k_steps_matrix - min_val) / (max_val - min_val)
                # config.optimize_tool['matrix'] = normalized_tensor
                # print("k_steps_idx:", k_steps_idx)
                # print("config.optimize_tool['matrix']:", config.optimize_tool['matrix'])
                lambda_k = config.optimize_tool['lambda']
                for each_k_idx in range(k_steps_matrix.shape[0]):
                    if each_k_idx > 0:
                        k_steps_matrix[each_k_idx] = lambda_k * k_steps_matrix[each_k_idx] + (1 - lambda_k) *k_steps_matrix[each_k_idx-1]
                config.optimize_tool['matrix'] = k_steps_matrix

                if dist.get_rank() > 0 and dist.get_rank() != 1:
                    dist.send(config.optimize_tool['lora_parm'], dst=1)
                    dist.send(config.optimize_tool['lora_grad'], dst=1)
                    dist.send(config.optimize_tool['matrix'], dst=1)

                
                if dist.get_rank() == 1:   # 需要根据实际指定 目前默认为rank为1的设备
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
                        # print("recv:", rand_idx)
                        
                    # print(f"rank:{dist.get_rank()}, matrix\n:{config.optimize_tool['matrix'][-1]}")
                    save_data_to_hdf5(config.optimize_tool)
                config.optimize_tool['input_x'].clear()
                return config.optimize_tool['matrix']

        _optimizer.zero_grad()  # TODO Order maters? or need it?

def optimizer_step(_loss, _optimizer, _model, _schedule, args, lora_info, is_update=True):
    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else:
        # params_before = {name: param.clone() for name, param in _model.named_parameters()}
        # grads_before_update = {name: param.grad.clone() if param.grad is not None else None for name, param in
        #                        _model.named_parameters()}
        if dist.get_rank() == 1:  # 需要根据实际指定 目前默认为rank为1的设备
            _loss.backward()
            first_hidden_states = lora_info[f'first_hidden_states_{dist.get_rank()}']

        # tensor_to_lora_list = lora_info[f"tensor_to_lora_list_{dist.get_rank()}"]
        combined_attn_temp_list = lora_info[f'combined_attn_temp_list_{dist.get_rank()}']

        for block_idx in reversed(range(len(combined_attn_temp_list))):
            
            # if isinstance(combined_attn_temp_list[block_idx], (int, float)) and math.isinf(combined_attn_temp_list[block_idx]):
            #     continue
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
                # print(f"==recv device:{dist.get_rank()}, grad:{combined_attn_temp_grad.shape}") # combined_attn_temp_grad.shape
                combined_attn_temp_list[block_idx].backward(combined_attn_temp_grad)
            # print(f"rank:{dist.get_rank()}, tensor_to_lora_list:{tensor_to_lora_list[0].grad}")
            # print("-"*50, block_idx, dist.get_rank())
        
        if dist.get_rank() == 1:
            first_hidden_states_grad, _ = dist_recv_grad(first_hidden_states.shape, torch.device('cuda', int(os.environ['LOCAL_RANK'])),
                                                        src=0, is_cal_criterion=config.optimize_tool['is_statistical_dict'], statistical_dict=config.statistical_dict)
            
            first_hidden_states.backward(first_hidden_states_grad)
        if config.optimize_tool['is_pred_net_data_collect']:
            # return update_model_by_optimizer_collect_data(_optimizer, _model, _schedule, args, is_update)
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
    # base_hidden_states_to_lora_list = base_info['base_hidden_states_to_lora_list']
    final_hidden_states_grad, _ = dist_recv_grad(final_hidden_states.shape, torch.device('cuda', int(os.environ['LOCAL_RANK'])),
                                                 src=1)

    final_hidden_states.backward(final_hidden_states_grad)

    # combined_attn_temp_list = base_info['base_combined_attn_temp_list']
    # for block_idx in reversed(range(len(combined_attn_temp_list))):
    #     print(f"combined_attn_temp_list:{len(combined_attn_temp_list[block_idx])}, grad?:{combined_attn_temp_list[block_idx].grad.shape}", )


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

            # _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk)
            if dist.get_rank() > 0:
                _lm_logits, _loss, lora_info = model(
                    _input, lm_labels=_target, lm_mask=_msk,  # label_smooth=args.label_smooth
                )
                # _lm_loss = _lm_loss.mean()
            else:
                lm_logits, presents, base_info = model(
                    data['input'].size(), lm_labels=args.device, lm_mask=None,  # label_smooth=args.label_smooth
                )
            if dist.get_rank() == 1:  # 需要根据实际指定 目前默认为rank为1的设备
                loss = _loss.mean()

                avg_lm_loss.update(loss.item())

                if idx % 100 == 0:
                    print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


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

    # train_loader.sampler.set_epoch(epoch)

    print("train_loader:", len(train_loader))
    # data = None
    # for batch in train_loader:
    #     data = batch
    #
    # if data is not None:
    #     data = {key: value for key, value in data.items()}
    #
    #     print("最后一组数据:", data['input'].shape)
    # exit()

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
                args.device)  # 注 torch.Size([4, 512])  ,including one whole line in train.jsonl(context and completion)
            _target = data['target'].to(args.device)  # 注 torch.Size([4, 512]) ,same as _input but no begin_token 3673
            _msk = data['mask'].to(
                args.device)  # 注 torch.Size([4, 512]) ,mask 1 begin form _target's completion , else 0

        if dist.get_rank() > 0:
            # print(f"_input:{_input[1, :]}")
            # print(f"rank:{dist.get_rank()}_target:{_target[1, :]}")
            _lm_logits, _lm_loss, lora_info = model(
                _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
            )
            if dist.get_rank() == 1: # 需要根据实际指定 目前默认为rank为1的设备
                _lm_loss = _lm_loss.mean()
            # print(f"_lm_logits:{_lm_logits[1, :]}")
            # print(f"rank:{dist.get_rank()}_lm_loss:{_lm_loss}")
        else:
            lm_logits, presents, base_info = model(
                data['input'].size(), lm_labels=args.device, lm_mask=None, label_smooth=args.label_smooth
            )

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        if dist.get_rank() > 0:
            if dist.get_rank() == 1:  # 需要根据实际指定 目前默认为rank为1的设备
                avg_lm_loss.update(_lm_loss.item())
            else:
                _lm_loss = 0
            # print(f"rank:{dist.get_rank()}, _lm_loss:{_lm_loss}")
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
                # decision_matrix_baseline = Pruning_baseline(k_steps_matrix, config.optimize_tool, kill_rate)
        elif dist.get_rank() == 0:
            base_optimizer_step(
                optimizer, model, scheduler, args, base_info,
                is_update=is_update,
            )

        # args.log_interval = 1
        if train_step % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | {idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

            if dist.get_rank() == 1:  # 需要根据实际指定 目前默认为rank为1的设备
                print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()

        # print(f"rank:{dist.get_rank()}, args.save_interval:{args.save_interval}")
        # if dist.get_rank() == 0:
        #     for n, p in model.named_parameters():
        #         print(f"name:{n}, parm:{p.size()}, grad?:{p.requires_grad}, rank:{dist.get_rank()}")
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
        # if idx == 100:
        #     exit()

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
    print("args.fp16:", args.fp16)

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

    # train_loader = DataLoader(
    #     train_data, batch_size=args.train_batch_size, num_workers=0,
    #     shuffle=False, pin_memory=False, drop_last=True,
    #     sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed)
    # )
    #
    # valid_loader = DataLoader(
    #     valid_data, batch_size=args.valid_batch_size, num_workers=0,
    #     shuffle=False, pin_memory=False, drop_last=False,
    #     sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed)
    # )

    # print("train_data_len:", train_data.num_batches)
    # k = 420  # 假设要选取前 10 个数据
    # selected_data = []
    # for i in range(min(k, len(train_data))):
    #     selected_data.append(train_data[i])
    # train_data = selected_data
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

    # optimize_matrix = np.ones((config.n_layer, config.n_head))
    cur_optimize_matrix = torch.randint(0, 2, size=(config.n_layer, config.n_head))
    # print("cur_optimize_matrix:", cur_optimize_matrix)
    cur_optimize_matrix = torch.zeros_like(cur_optimize_matrix)
    config.optimize_tool['is_statistical_dict'] = False  #  是否进行预实验的统计，默认关闭
    config.optimize_tool['is_pred_net_data_collect'] = False  #  是否进行预测网络数据集收集，默认关闭
    config.optimize_tool['is_include_input_x'] = False  #  是否进行保存网络输入embedding，默认关闭
    config.optimize_tool['is_cal_s_comp'] = False  #  是否s_comp公式的计算以及剪枝，默认关闭
    # cur_optimize_matrix[0:int(config.n_layer*1/3),:] = 1
    config.optimize_tool['cur_step_decision_matrix'] = cur_optimize_matrix  # 最终优化得到的决策矩阵
    config.optimize_tool['vector'] = None
    config.optimize_tool['head_alloc_device'] = [8,5,3]  # 每个设备静态的分几个head 预先设置 [9,4,3] [11,5] [16] / [6,5,5] [8,8] / [8,5,3] [10,6,4]
    head_device_idx_list = [0] * (len(config.optimize_tool['head_alloc_device'])+1)
    start_head = 0
    for device_idx, device_num in enumerate(config.optimize_tool['head_alloc_device']):
        end_head = start_head + device_num
        start_head = end_head
        head_device_idx_list[device_idx+1] = start_head
    config.optimize_tool['head_device_idx_list'] = head_device_idx_list
    

    config.optimize_tool['hidden_state_shape'] = torch.Size((args.train_batch_size, args.seq_len, config.n_embd))
    print("config.optimize_tool['hidden_state_shape']:", config.optimize_tool['hidden_state_shape'])  
    config.optimize_tool['history_k_steps'] = 10
    config.optimize_tool['lambda'] = 0.8  # np.arange(0.05, 1, 0.05)
    config.optimize_tool['alpha'] = 1
    config.optimize_tool['rho'] = 0.2
    config.optimize_tool['beta'] = 0.3
    config.optimize_tool['tau'] = 0.7  # 当前的含义是保留多少%的分数
    config.optimize_tool['eta'] = 0.8  # 跳层的阈值参数 0.4
    config.optimize_tool['head_pruned_device'] = config.optimize_tool['head_alloc_device'] * 0# [round(item * config.optimize_tool['beta']) for item in config.optimize_tool['head_alloc_device']]
    optimize_matrix = torch.zeros((config.optimize_tool['history_k_steps'], config.n_layer, config.n_head), dtype=torch.float, device=args.device)
    config.optimize_tool['matrix'] = optimize_matrix  # 分数矩阵
    config.optimize_tool['decision_matrix'] = torch.zeros_like(optimize_matrix)  # 最终优化得到的决策矩阵

    # 以下是组成预测网络数据集的相关训练数据
    config.optimize_tool['input_x'] = []
    config.optimize_tool['base_parm'] = None
    config.optimize_tool['lora_parm'] = None
    config.optimize_tool['lora_grad'] = torch.zeros_like(optimize_matrix)
    config.optimize_tool['epoch'] = None
    config.optimize_tool['dataset_path'] = './custom_data'
    vis_save_path = None
    if dist.get_rank() > 0:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        vis_save_path = os.path.join(args.work_dir, f'{current_time}')
    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print(f'loading model pretrained weight.rank:{args.dist.get_rank()}')
        lm_net.load_weight(torch.load(args.init_checkpoint))

    # lm_net = lm_net.cuda()
    lm_net = lm_net.to(args.local_rank)

    # for n, p in lm_net.named_parameters():
    #     print(f"name:{n}, parm:{p.size()}")
    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)
    optimizer = create_adam_optimizer_from_args(lm_net, args)

    if args.max_step is None:
        # args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        local_world_size = 1
        args.max_step = (args.max_epoch * train_data.num_batches + local_world_size - 1) // local_world_size
        # args.max_step = (args.max_epoch * k/3 + local_world_size - 1) // local_world_size
        print('set max_step:', args.max_step)

    scheduler = create_optimizer_scheduler(optimizer, args)
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
    # lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)

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
                # res = config.statistical_dict['send_data_array_per_epoch_timestep_mi']
                # print("res:", len(res))
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
