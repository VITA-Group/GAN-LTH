import pdb
import pickle
import numpy as np
import torch
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


