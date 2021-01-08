# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functions import validate_cp, calculate_metrics, create_dir
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os, argparse
import numpy as np
from skimage.io import imsave

from models.sngan_cifar10 import Generator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def calculate_IS_FID(G):
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # fid stat
    fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    assert os.path.exists(fid_stat)

    # IS and FID:
    inception_score, fid_score = \
            calculate_metrics(args.fid_buffer_dir, args.num_eval_imgs, args.eval_batch_size, args.latent_dim, 
                            fid_stat, G, do_IS=args.do_IS, do_FID=args.do_FID)
    print('Inception score: %s, FID score: %s.' % (inception_score, fid_score))

def save_random_z():
    random_z = np.random.normal(0, 1, (64, args.latent_dim))
    np.save('random_z.npy', random_z)

def gen_img(G):
    # initial
    if not os.path.isfile('random_z.npy'):
        save_random_z()
    fixed_z = torch.cuda.FloatTensor(np.load('random_z.npy'))
    # generate imgs:
    imgs = validate_cp(fixed_z, G, n_row=8)

    return imgs # [0,1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='7', help='which gpu to use')

    parser.add_argument('--gf_dim', type=int, default=256, help='The base channel num of gen')
    parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
    parser.add_argument('--bottom_width', type=int, default=4, help="the base resolution of the GAN")

    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--num_eval_imgs', type=int, default=50000)

    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--load-epoch', type=int, default=100)
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    ###
    # for pruning models:
    exp_str = args.dir
    test_result_dir = os.path.join('output', exp_str, 'Samples')
    create_dir(test_result_dir)
    args.fid_buffer_dir = os.path.join('output', exp_str, 'fid_buffer')
    args.load_path = os.path.join('output', exp_str, 'pth', 'epoch%d.pth' % args.load_epoch)
    
    # # for original model:
    # exp_str = 'sngan_cifar10_2019_10_24_12_19_30'
    # test_result_dir = os.path.join('logs', exp_str, 'Samples')
    # create_dir(test_result_dir)
    # args.fid_buffer_dir = os.path.join('logs', exp_str, 'fid_buffer')
    # args.load_path = os.path.join('logs', exp_str, 'Model', 'checkpoint_best.pth')
    ###

    args.do_IS = True
    args.do_FID = True


    # state dict:
    assert os.path.exists(args.load_path)
    checkpoint = torch.load(args.load_path)
    print('=> loaded checkpoint %s' % args.load_path)
    if 'output' in args.load_path:
        state_dict = checkpoint['generator']
    elif 'logs' in args.load_path:
        state_dict = checkpoint['avg_gen_state_dict']
    
    # define network
    G = Generator(bottom_width=args.bottom_width, gf_dim=args.gf_dim, latent_dim=args.latent_dim).cuda()
    for p in G.parameters():
        p.requires_grad = False
    # load pth
    G.load_state_dict(state_dict)

    calculate_IS_FID(G)

    imgs = gen_img(G)
    imsave(os.path.join(test_result_dir, 'test_result.png'), imgs)
