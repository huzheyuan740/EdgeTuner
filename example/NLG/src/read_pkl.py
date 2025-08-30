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

torch.set_printoptions(threshold=100000)
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

if __name__ == '__main__':
    for epoch_idx in range (1, 6, 1):
        path = f'../NLG/trained_models/GPT2_M/e2e_all_trainset/20250328121021/epoch_{epoch_idx}/save_dist_file_14020.pkl'#'../NLG/trained_models/GPT2_M/e2e_all_trainset/20250328233633/epoch_1/save_dist_file_14020.pkl'
        load_dict = read_vis_file(path)
        folder_path = os.path.dirname(path)
        for key, value in load_dict.items():
            print((key))
            draw_each_catagory_pic_for_timestep_heatmap(load_dict, key, folder_path)
        print(load_dict['send_data_array_per_epoch_timestep_mi'][0].shape)
        print(len(load_dict['send_data_array_per_epoch_timestep_mi']))
