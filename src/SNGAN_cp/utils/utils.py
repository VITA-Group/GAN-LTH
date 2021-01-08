# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import os
import torch
import dateutil.tz
from datetime import datetime
import time
import logging

count_ops = 0
num_ids = 0
def get_feature_hook(self, _input, _output):
	global count_ops, num_ids 
	print('------>>>>>>')
	print('{}th node, input shape: {}, output shape: {}, input channel: {}, output channel {}'.format(
		num_ids, _input[0].size(2), _output.size(2), _input[0].size(1), _output.size(1)))
	print(self)
	delta_ops = self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] * _output.size(2) * _output.size(3) / self.groups
	count_ops += delta_ops
	print('ops is {:.6f}M'.format(delta_ops / 1024.  /1024.))
	num_ids += 1
	print('')

def measure_model(net, z_dim):
	import torch
	import torch.nn as nn
	_input = torch.randn((1, z_dim))
	#_input, net = _input.cpu(), net.cpu()
	hooks = []
	for module in net.named_modules():
		if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.ConvTranspose2d):
			# print(module)
			hooks.append(module[1].register_forward_hook(get_feature_hook))

	_out = net(_input)
	global count_ops
	print('count_ops: {:.6f}M'.format(count_ops / 1024. /1024.)) # in Million

def layer_param_num(model, param_name=['weight']):
	count_res = {}
	for name, W in model.named_parameters():
		if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
			# W_nz = torch.nonzero(W.data)
			W_nz = torch.flatten(W.data)
			if W_nz.dim() > 0:
				count_res[name] = W_nz.shape[0]
	return count_res

def model_param_num(model, param_name=['weight']):
	layer_size_dict = layer_param_num(model, param_name)
	return sum(layer_size_dict.values()) / 1024 / 1024

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def save_checkpoint_cp(states, output_dir, filename):
    torch.save(states, os.path.join(output_dir, filename))

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import os
import torch
import dateutil.tz
from datetime import datetime
import time
import logging

import pdb
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


def pruning_generate(model,px):

    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def see_remain_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))     
    print('remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
def see_remain_rate_orig(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            sum_list = sum_list+float(m.weight_orig.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight_orig == 0))     
    print('remain weight = ', 100*(1-zero_sum/sum_list),'%')


def rewind_weight(model_dict, target_model_dict_keys):

    new_dict = {}
    for key in target_model_dict_keys:
        if 'mask' not in key:
            if 'orig' in key:
                ori_key = key[:-5]
            else:
                ori_key = key 
            new_dict[key] = model_dict[ori_key]

    return new_dict

def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))
