#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import cka

import seaborn as sns
import matplotlib.pyplot as plt
import pickle



def kl_divergence(p, q):
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def mutual_information(p, q):
    feature_shape = p.shape[-1]
    p_for_mi = p.view(-1, feature_shape)
    q_for_mi = q.view(-1, feature_shape)
    result = cka.cka(cka.gram_linear(p_for_mi.float()), cka.gram_linear(q_for_mi.float()))
    if torch.isnan(result):
        result = torch.tensor(0.0, dtype=p_for_mi.dtype, device=p_for_mi.device)
    return result


def draw_each_catagory_pic(statistical_dict, key, path, group_size=None):
    data = statistical_dict[key]
    assert group_size is None
    if group_size is None:
        group_size = len(data)

    num_groups = int(len(data) / group_size)
    vis_path = path
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    for i in range(num_groups):
        start_index = i * group_size
        end_index = (i + 1) * group_size
        group_data = data[start_index:end_index]
        
        group_array = np.array(group_data).T
        
        plt.figure() 
        sns.heatmap(group_array, cmap='hot')
        plt.title(f'Heatmap of group {i + 1}')
        plt.xlabel('Step Index')
        plt.ylabel('Block Index')
        plt.tight_layout()

        # filename = f"{start_index}-{end_index - 1}.jpg"
        filename = f"{key}.jpg"
        full_path = os.path.join(vis_path, filename)

        # 保存热力图
        plt.savefig(full_path)
        plt.close()

    print("all heatmap saved")


def draw_each_catagory_pic_for_square_heatmap(statistical_dict, key, path, step_idx=None):
    data = statistical_dict[key]

    vis_path = path
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    num_time_steps = 100  # vis time steps
    step_interval = int(step_idx / num_time_steps)
    selected_time_steps = np.arange(0, num_time_steps * step_interval, step_interval)

    rows = int(np.sqrt(num_time_steps))
    cols = int(np.ceil(num_time_steps / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    first_ax_img = None

    # draw heatmap
    for i, time_step in enumerate(selected_time_steps):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        sns.heatmap(data[time_step], cmap='hot', ax=ax, cbar=False, annot=False)
        ax.set_title(f'{time_step}')
        ax.axis('off')
        if i == 0:
            first_ax_img = ax.collections[0]

    # 调整子图布局，增加右边距
    left_margin = 0.05
    right_margin = 0.90 
    bottom_margin = 0.05
    top_margin = 0.95
    wspace = 0 
    hspace = 0  
    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)

    cbar_width = 0.03
    cbar_left = right_margin + 0.02 
    cbar_bottom = 0.1
    cbar_height = 0.8

    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
   
    fig.colorbar(first_ax_img, cax=cbar_ax)

    filename = f"{key}.jpg"
    full_path = os.path.join(vis_path, filename)

    plt.savefig(full_path)

    plt.close()


def draw_each_catagory_pic_for_timestep_heatmap(statistical_dict, key, path, step_idx=None):
    data = statistical_dict[key]
    step_idx = len(data)

    vis_path = path
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    num_time_steps = 100 
    step_interval = int(step_idx / num_time_steps)
    selected_time_steps = np.arange(0, num_time_steps * step_interval, step_interval)
    print("selected_time_steps:", selected_time_steps)

    rows = int(np.sqrt(num_time_steps))
    cols = int(np.ceil(num_time_steps / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    first_ax_img = None

    for i, time_step in enumerate(selected_time_steps):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        sns.heatmap(data[time_step], cmap='hot', ax=ax, cbar=False, annot=False)

        ax.set_title(f'{time_step}')
        ax.set(yticks=[])
        ax.axis('off')
        # 记录第一个子图的 AxesImage 对象
        if i == 0:
            first_ax_img = ax.collections[0]

    left_margin = 0.05
    right_margin = 0.90
    bottom_margin = 0.05 
    top_margin = 0.95  
    wspace = 0  
    hspace = 0  
    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)

    cbar_width = 0.03
    cbar_left = right_margin + 0.02
    cbar_bottom = 0.1
    cbar_height = 0.8

    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    fig.colorbar(first_ax_img, cax=cbar_ax)

    filename = f"{key}.jpg"
    full_path = os.path.join(vis_path, filename)

    plt.savefig(full_path)

    plt.close()


def update_clear_statistical_dict(statistical_dict, step_key, epoch_key, period_window=None, periodic_clear_epoch_key=False):
    if statistical_dict.get(step_key) is not None:
        if statistical_dict.get(epoch_key) is None:
            statistical_dict[epoch_key] = []
        statistical_dict[epoch_key].append(statistical_dict[step_key].copy())
        if not periodic_clear_epoch_key:
            statistical_dict[step_key].clear()
            statistical_dict[step_key] = None


def calculate_info_matrix_time_step(statistical_dict, step_key, epoch_key, calculate_in_out_info_flag, period_window, device=None):
    if statistical_dict.get(epoch_key) is not None:
        origin_list = statistical_dict[epoch_key]
        assert len(origin_list) == period_window
        block_num = len(origin_list[0])
        kl_matrix = np.zeros((block_num, block_num))
        start_time = time.time()
        mi_matrix = np.zeros((block_num, period_window))
        for k in range(block_num):
            for i in range(period_window):
                # calculate mutual_information
                mi = mutual_information(origin_list[i][k].to(device), origin_list[period_window-1][k].to(device))
                mi_matrix[k, i] = mi.item()
        print(f"{epoch_key} time::{time.time() - start_time}")
        mi_epoch_key = epoch_key + '_timestep_mi'
        if statistical_dict.get(mi_epoch_key) is None:
            statistical_dict[mi_epoch_key] = []
        statistical_dict[mi_epoch_key].append(mi_matrix.copy())

        statistical_dict[epoch_key].clear()
        statistical_dict[epoch_key] = None


def calculate_info_matrix(statistical_dict, step_key, epoch_key, calculate_in_out_info_flag, device=None):
    if statistical_dict.get(step_key) is not None:
        prob_list = [F.softmax(tensor.to(device), dim=-1) for tensor in statistical_dict[step_key]]
        origin_list = [tensor.to(device) for tensor in statistical_dict[step_key]]
        block_num = len(statistical_dict[step_key])
        kl_matrix = np.zeros((block_num, block_num))
        mi_matrix = np.zeros((block_num, block_num))
        begin_time = time.time()
        for i in range(block_num):
            for j in range(block_num):
                if True:
                    mi = mutual_information(origin_list[i], origin_list[j])
                    mi_matrix[i, j] = mi.item()
        mi_epoch_key = epoch_key + '_mi'
        if statistical_dict.get(mi_epoch_key) is None:
            statistical_dict[mi_epoch_key] = []
        statistical_dict[mi_epoch_key].append(mi_matrix.copy())
        if not calculate_in_out_info_flag:
            statistical_dict[step_key].clear()
            statistical_dict[step_key] = None


def calculate_in_out_info_matrix(statistical_dict, step_key_in, step_key_out, epoch_key):
    if statistical_dict.get(step_key_in) is not None and len(statistical_dict[step_key_in]):
        origin_copy_list_in = [torch.cat([tensor] * 3, dim=-1) for tensor in statistical_dict[step_key_in]]
        prob_list_in = [torch.cat([F.softmax(tensor, dim=-1)] * 3, dim=-1) for tensor in statistical_dict[step_key_in]]
        prob_list_out = [F.softmax(tensor, dim=-1) for tensor in statistical_dict[step_key_out]]
        block_num = len(statistical_dict[step_key_in])
        kl_matrix = np.zeros((block_num))
        mi_matrix = np.zeros((block_num))
        for i in range(block_num):
            # calculate kl-divergence
            kl = kl_divergence(prob_list_in[i], prob_list_out[i])
            kl_matrix[i] = torch.mean(kl).item()

            # calculate mutual information
            mi = mutual_information(origin_copy_list_in[i], statistical_dict[step_key_out][i])
            mi_matrix[i] = mi.item()
        kl_epoch_key = epoch_key+'_kl'
        if statistical_dict.get(kl_epoch_key) is None:
            statistical_dict[kl_epoch_key] = []
        statistical_dict[kl_epoch_key].append(kl_matrix.copy())
        mi_epoch_key = epoch_key + '_mi'
        if statistical_dict.get(mi_epoch_key) is None:
            statistical_dict[mi_epoch_key] = []
        statistical_dict[mi_epoch_key].append(mi_matrix.copy())
        statistical_dict[step_key_in].clear()
        statistical_dict[step_key_in] = None
        statistical_dict[step_key_out].clear()
        statistical_dict[step_key_out] = None


def get_statistical_dict_key_list(statistical_dict, key, data):
    data_save = data.detach().cpu()#.numpy()
    if statistical_dict.get(key) is not None:
        statistical_dict[key].append(data_save)
    else:
        statistical_dict[key] = []
        statistical_dict[key].append(data_save)


def save_vis_file(statistical_dict):
    path = statistical_dict['vis_save_path']
    print(f"path:{path}, epoch:{str(statistical_dict['epoch'])}")
    vis_save_path = os.path.join(path, 'epoch_' + str(statistical_dict['epoch']))
    vis_path = vis_save_path
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    filename = f"save_dist_file_{statistical_dict['idx']}.pkl"
    full_path = os.path.join(vis_path, filename)
    save_dist = {}
    for key, value in statistical_dict.items():
        if 'timestep_' in key:
            save_dist[key] = value

    with open(full_path, 'wb') as f:
        pickle.dump(save_dist, f)


def read_vis_file(path):
    with open(path, 'rb') as f:
        load_dict = pickle.load(f)
    return load_dict


def save_vis_pic(statistical_dict, group_size=None, step_idx=None):
    path = statistical_dict['vis_save_path']
    print(f"path:{path}, epoch:{str(statistical_dict['epoch'])}")
    vis_save_path = os.path.join(path, 'epoch_' + str(statistical_dict['epoch']))

    # draw_each_catagory_pic(statistical_dict, 'send_data_entropy_logic_per_epoch', vis_save_path, group_size)
    # draw_each_catagory_pic(statistical_dict, 'send_grad_entropy_logic_per_epoch', vis_save_path, group_size)
    # draw_each_catagory_pic(statistical_dict, 'recv_data_entropy_logic_per_epoch', vis_save_path, group_size)
    # draw_each_catagory_pic(statistical_dict, 'recv_grad_entropy_logic_per_epoch', vis_save_path, group_size)

    # draw_each_catagory_pic_for_square_heatmap(statistical_dict, 'send_data_array_per_epoch_kl', vis_save_path, step_idx)
    draw_each_catagory_pic_for_square_heatmap(statistical_dict, 'send_data_array_per_epoch_mi', vis_save_path, step_idx)
    # draw_each_catagory_pic_for_square_heatmap(statistical_dict, 'send_grad_array_per_epoch_kl', vis_save_path, step_idx)
    draw_each_catagory_pic_for_square_heatmap(statistical_dict, 'send_grad_array_per_epoch_mi', vis_save_path, step_idx)
    # draw_each_catagory_pic_for_square_heatmap(statistical_dict, 'recv_data_array_per_epoch_kl', vis_save_path, step_idx)
    draw_each_catagory_pic_for_square_heatmap(statistical_dict, 'recv_data_array_per_epoch_mi', vis_save_path, step_idx)
    # draw_each_catagory_pic_for_square_heatmap(statistical_dict, 'recv_grad_array_per_epoch_kl', vis_save_path, step_idx)
    draw_each_catagory_pic_for_square_heatmap(statistical_dict, 'recv_grad_array_per_epoch_mi', vis_save_path, step_idx)

    # draw_each_catagory_pic(statistical_dict, 'in_out_data_array_per_epoch_kl', vis_save_path, group_size)
    # draw_each_catagory_pic(statistical_dict, 'in_out_data_array_per_epoch_mi', vis_save_path, group_size)
    # draw_each_catagory_pic(statistical_dict, 'in_out_grad_array_per_epoch_kl', vis_save_path, group_size)
    # draw_each_catagory_pic(statistical_dict, 'in_out_grad_array_per_epoch_mi', vis_save_path, group_size)


def dist_send_data(data, dst, ref_event=None, is_cal_criterion=False, statistical_dict=None, is_belong_block=False,
                   is_backward=False):
    if is_cal_criterion and ref_event is not None:
        # start_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        #
        # torch.cuda.synchronize()
        # start_time = ref_event.elapsed_time(start_event)
        start_time = time.time() - ref_event
        timestamp_tensor = torch.tensor([start_time], dtype=torch.float32).to(data.device)

        # 合并时间戳和多维 tensor
        combined_tensor = torch.cat((timestamp_tensor, data.view(-1)), dim=0)
    elif is_cal_criterion:
        assert statistical_dict is not None
        memory_logic = data.nbytes / (1024 ** 2)
        memory_cuda = torch.cuda.memory_allocated(data.device) / (1024 ** 2)

        if is_belong_block:
            if is_backward:
                get_statistical_dict_key_list(statistical_dict, 'send_grad_array_per_step', data)
            else:
                get_statistical_dict_key_list(statistical_dict, 'send_data_array_per_step', data)

        combined_tensor = data
    else:
        combined_tensor = data
    dist.send(tensor=combined_tensor, dst=dst)


def dist_recv_data(data_shape, device, src, ref_event=None, is_cal_criterion=False, statistical_dict=None,
                   is_belong_block=False, is_backward=False):
    if is_cal_criterion and ref_event is not None:
        data_combined = torch.empty(1 + torch.prod(torch.tensor(data_shape)).item()).to(device)
        dist.recv(tensor=data_combined, src=src)
        start_time = data_combined[0].item()
        data = data_combined[1:].view(data_shape)
        data.requires_grad_(True)
        end_time = time.time() - ref_event
        latency = end_time - start_time
    elif is_cal_criterion:
        data = torch.empty(tuple(data_shape)).to(device)
        data.requires_grad_(True)
        dist.recv(tensor=data, src=src)
        latency = None
        assert statistical_dict is not None
        memory_logic = data.nbytes / (1024 ** 2)
        memory_cuda = torch.cuda.memory_allocated(data.device) / (1024 ** 2)
        if is_belong_block:
            assert is_backward is False
            get_statistical_dict_key_list(statistical_dict, 'recv_data_array_per_step', data)
    else:
        data = torch.empty(tuple(data_shape)).to(device)
        data.requires_grad_(True)
        dist.recv(tensor=data, src=src)
        latency = None

    return data, latency


def dist_recv_grad(data_shape, device, src, ref_event=None, is_cal_criterion=False, statistical_dict=None,
                   is_belong_block=False, is_backward=False):
    if is_cal_criterion and ref_event is not None:
        data_combined = torch.empty(1 + torch.prod(torch.tensor(data_shape)).item()).to(device)
        dist.recv(tensor=data_combined, src=src)
        start_time = data_combined[0].item()
        data = data_combined[1:].view(data_shape)
        end_time = time.time() - ref_event
        latency = end_time - start_time
    elif is_cal_criterion:
        data = torch.empty(tuple(data_shape)).to(device)
        dist.recv(tensor=data, src=src)
        latency = None
        assert statistical_dict is not None
        if is_belong_block:
            assert is_backward is True
            if is_backward:
                get_statistical_dict_key_list(statistical_dict, 'recv_grad_array_per_step', data)
    else:
        data = torch.empty(tuple(data_shape)).to(device)
        dist.recv(tensor=data, src=src)
        latency = None

    return data, latency
