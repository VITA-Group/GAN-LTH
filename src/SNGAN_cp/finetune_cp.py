import cfg
import models
from models.sngan_cifar10 import Generator, Discriminator
import datasets
import random
from functions import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def get_param(name, state_dict, in_channel, out_channel):
    return (state_dict[name + '.weight'][out_channel, :, :, :][:, in_channel, :, :], state_dict[name + '.bias'][out_channel])

def load_subnet(args, state_dict, init_state_dict):

    # import network
    G = Generator(bottom_width=args.bottom_width, gf_dim=args.gf_dim, latent_dim=args.latent_dim)
    G.load_state_dict(state_dict)

    G0 = Generator(bottom_width=args.bottom_width, gf_dim=args.gf_dim, latent_dim=args.latent_dim)
    G0.load_state_dict(init_state_dict)


    # extract:
    dim_lst, selected_input_channel_idx_lst = [], []
    for name, m in G.named_modules():
        if isinstance(m, nn.BatchNorm2d) and m.weight is not None:
            gamma = m.weight.data.detach().cpu().numpy()
            none_zero_dim = np.sum(gamma!=0)
            dim_lst.append(none_zero_dim)
            selected_idx = np.where(gamma!=0)[0]
            selected_input_channel_idx_lst.append(selected_idx)
            # print(name)
    print('dim_lst:', dim_lst, len(dim_lst))
    print('selected_input_channel_idx_lst:', len(selected_input_channel_idx_lst))
    assert len(dim_lst) == 7 # for cartoongan generator
    
    # get hidden_dim_lst:
    hidden_dim_lst = dim_lst[1:6:2] # 1,3,5

    # selected_input_channel_idx_lst:
    selected_input_channel_idx_lst_all = selected_input_channel_idx_lst
    selected_input_channel_idx_lst = selected_input_channel_idx_lst[0::2] # 0,2,4,6

    # define subnet:
    G_sub = Generator(bottom_width=args.bottom_width, gf_dim=args.gf_dim, latent_dim=args.latent_dim,
            hidden_dim_lst=hidden_dim_lst, selected_input_channel_idx_lst=selected_input_channel_idx_lst)

    # measure sub model
    print(dim_lst)
    print(selected_input_channel_idx_lst)
    
    weight_list = {'block2.c1': get_param('block2.c1', init_state_dict, selected_input_channel_idx_lst_all[0], selected_input_channel_idx_lst_all[1]),
                   'block2.c2': get_param('block2.c2', init_state_dict, selected_input_channel_idx_lst_all[1], list(range(256))),
                   'block2.b2': (init_state_dict['block2.b2.weight'][selected_input_channel_idx_lst_all[1]], init_state_dict['block2.b2.bias'][selected_input_channel_idx_lst_all[1]]),
                   'block3.c1': get_param('block3.c1', init_state_dict, selected_input_channel_idx_lst_all[2], selected_input_channel_idx_lst_all[3]),
                   'block3.c2': get_param('block3.c2', init_state_dict, selected_input_channel_idx_lst_all[3], list(range(256))),
                   'block3.b2': (init_state_dict['block3.b2.weight'][selected_input_channel_idx_lst_all[3]], init_state_dict['block3.b2.bias'][selected_input_channel_idx_lst_all[3]]),
                   'block4.c1': get_param('block4.c1', init_state_dict, selected_input_channel_idx_lst_all[4], selected_input_channel_idx_lst_all[5]),
                   'block4.c2': get_param('block4.c2', init_state_dict, selected_input_channel_idx_lst_all[5], list(range(256))),
                   'block4.b2': (init_state_dict['block4.b2.weight'][selected_input_channel_idx_lst_all[5]], init_state_dict['block2.b2.bias'][selected_input_channel_idx_lst_all[5]]),
                   'c5': get_param('c5', init_state_dict, selected_input_channel_idx_lst_all[6], list(range(3)))}
    
    print(weight_list.keys())
    for name, m in G_sub.named_modules():
        if name in weight_list:
            m.weight.data.copy_(weight_list[name][0])
            m.bias.data.copy_(weight_list[name][1])
    
    return G_sub


def main():
    args = cfg.parse_args()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net = Generator(bottom_width=args.bottom_width, gf_dim=args.gf_dim, latent_dim=args.latent_dim).cuda()
    dis_net = eval('models.'+args.model+'.Discriminator')(args=args).cuda()
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    
    initial_gen_net_weight = gen_net.state_dict()
    initial_dis_net_weight = dis_net.state_dict()
    
    init_state_dict = gen_net.state_dict()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    exp_str = args.dir
    args.load_path = os.path.join('output',
        exp_str,
        'pth', 'epoch{}.pth'.format(args.load_epoch))

    # state dict:
    assert os.path.exists(args.load_path)
    checkpoint = torch.load(args.load_path)
    print('=> loaded checkpoint %s' % args.load_path)
    state_dict = checkpoint['generator']
    gen_net = load_subnet(args, state_dict, deepcopy(state_dict)).cuda()
    avg_gen_net = deepcopy(gen_net)
    
    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    else:
        raise NotImplementedError('no fid stat for %s' % args.dataset.lower())
    assert os.path.exists(fid_stat)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    np.random.seed(args.random_seed)
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    
    start_epoch = 0
    best_fid = 1e4
    
    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    #logger.info('=> loaded checkpoint %s (epoch %d)' % (checkpoint_file, start_epoch))

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    gen_avg_param = copy_params(gen_net)
    # train loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
              lr_schedulers)

        if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            inception_score, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict, epoch)
            logger.info('Inception score: %.4f, FID score: %.4f || @ epoch %d.' % (inception_score, fid_score, epoch))
            load_params(gen_net, backup_param)
            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
            else:
                is_best = False
        else:
            is_best = False

        avg_gen_net.load_state_dict(gen_net.state_dict())
        load_params(avg_gen_net, gen_avg_param)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper
        }, is_best, args.path_helper['ckpt_path'])


if __name__ == '__main__':
    main()
