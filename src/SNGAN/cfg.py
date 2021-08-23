# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max-epoch',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--max-iter',
        type=int,
        default=50000,
        help='set the max iteration number')
    parser.add_argument(
        '-gen-bs',
        '--gen-batch-size',
        type=int,
        default=128,
        help='size of the batches')
    parser.add_argument(
        '-dis-bs',
        '--dis-batch-size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '--g-lr',
        type=float,
        default=0.0002,
        help='adam: gen learning rate')
    parser.add_argument(
        '--d-lr',
        type=float,
        default=0.0002,
        help='adam: disc learning rate')
    parser.add_argument(
        '--lr-decay',
        action='store_true',
        help='learning rate decay or not')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=128,
        help='dimensionality of the latent space')
    parser.add_argument(
        '--img-size',
        type=int,
        default=32,
        help='size of each image dimension')
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='number of image channels')
    parser.add_argument(
        '--n-critic',
        type=int,
        default=5,
        help='number of training steps for discriminator per iter')
    parser.add_argument(
        '--val-freq',
        type=int,
        default=20,
        help='interval between each validation')
    parser.add_argument(
        '--print-freq',
        type=int,
        default=50,
        help='interval between each verbose')
    parser.add_argument(
        '--load-path',
        type=str,
        help='The reload model path')
    parser.add_argument(
        '--exp-name',
        type=str,
        help='The name of exp')
    parser.add_argument(
        '--d-spectral_norm',
        type=str2bool,
        default=True,
        help='add spectral_norm on discriminator?')
    parser.add_argument(
        '--g-spectral_norm',
        type=str2bool,
        default=False,
        help='add spectral_norm on generator?')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='dataset type')
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument('--init-type', type=str, default='xavier_uniform',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    parser.add_argument('--gf-dim', type=int, default=256,
                        help='The base channel num of gen')
    parser.add_argument('--df-dim', type=int, default=128,
                        help='The base channel num of disc')
    parser.add_argument(
        '--model',
        type=str,
        default='sngan_cifar10',
        help='path of model')
    parser.add_argument('--eval-batch-size', type=int, default=100)
    parser.add_argument('--num-eval-imgs', type=int, default=50000)
    parser.add_argument(
        '--bottom-width',
        type=int,
        default=4,
        help="the base resolution of the GAN")
    parser.add_argument(
        '--pruning-method',
        type=str,
        default='l1',
        help="the pruning method of G")
    parser.add_argument('--random_seed', type=int, default=1)
    # Baseline Group
    parser.add_argument('--reset-dis-net', action="store_true")
    parser.add_argument('--percent', type=float, help="prune percent")
    parser.add_argument('--second-seed', action="store_true")
    parser.add_argument('--fix-G', action="store_true")
    parser.add_argument('--load-D-optimizer', action="store_true")
    parser.add_argument('--finetune-D', action="store_true")
    parser.add_argument('--finetune-G', action="store_true")
    parser.add_argument('--use-kd-D', action="store_true")
    parser.add_argument('--resume', type=str, default=None)
    
    parser.add_argument('--save-path', type=str, default='initial_weights')
    parser.add_argument('--init-path', type=str, default=None)
    parser.add_argument('--rewind-path', type=str, default=None)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default="run")

    opt = parser.parse_args()
    return opt
