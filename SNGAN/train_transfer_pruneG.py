import cfg
import models
import datasets
import random
from functions import train, train_with_mask, train_with_mask_kd, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger,  see_remain_rate, see_remain_rate_orig, rewind_weight
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from collections import OrderedDict
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def pruning_generate(model, state_dict):

    parameters_to_prune =[]
    for (name, m) in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m = prune.custom_from_mask(m, name = 'weight', mask = state_dict[name + ".weight_mask"])


def main():
    args = cfg.parse_args()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import netwo

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

    gen_net = eval('models.'+args.model+'.Generator')(args=args).cuda()
    dis_net = eval('models.'+args.model+'.Discriminator')(args=args).cuda()
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    avg_gen_net = deepcopy(gen_net)
    initial_gen_net_weight = torch.load(os.path.join(args.init_path, 'initial_gen_net.pth'), map_location="cpu")
    initial_dis_net_weight = torch.load(os.path.join(args.init_path, 'initial_dis_net.pth'), map_location="cpu")
    assert id(initial_dis_net_weight) != id(dis_net.state_dict())
    assert id(initial_gen_net_weight) != id(gen_net.state_dict())
    
    
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
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/fid_stats_stl10_train.npz'
    else:
        raise NotImplementedError('no fid stat for %s' % args.dataset.lower())
    assert os.path.exists(fid_stat)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    
    start_epoch = 0
    best_fid = 1e4

    print('=> resuming from %s' % args.load_path)
    assert os.path.exists(args.load_path)
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    
    pruning_generate(gen_net, checkpoint['gen_state_dict'])
    pruning_generate(avg_gen_net, checkpoint['gen_state_dict'])
    see_remain_rate(gen_net)
    
    if not args.finetune_G:
        gen_weight = gen_net.state_dict()
        gen_orig_weight = rewind_weight(initial_gen_net_weight, gen_weight.keys())
        gen_weight.update(gen_orig_weight)
        gen_net.load_state_dict(gen_weight)
    gen_avg_param = copy_params(gen_net)
        
    if args.finetune_D:
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
    else:
        dis_net.load_state_dict(initial_dis_net_weight)  
    
    
    args.path_helper = set_log_dir('logs', args.exp_name + "_{}".format(args.percent))
    logger = create_logger(args.path_helper['log_path'])
    #logger.info('=> loaded checkpoint %s (epoch %d)' % (checkpoint_file, start_epoch))

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        see_remain_rate(gen_net)

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
