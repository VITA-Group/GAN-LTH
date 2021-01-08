import cfg
from models.sngan_cifar10 import Generator, Discriminator
import datasets
from functions import VGGFeature, gram_matrix, validate_cp, soft_sign, show_sparsity, create_dir, LambdaLR, fourD2threeD, calculate_metrics
from utils.utils import save_checkpoint_cp
from utils.inception_score import _init_inception, get_inception_score
from utils.fid_score import create_inception_graph, check_or_download_inception, calculate_fid_given_paths

import os, time
import numpy as np
from skimage.io import imsave
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import make_grid

# args:
args = cfg.parse_args()

# dir:
opt_str = 'adam_lr%s_sgd_lr%s_epoch%d_de%d_batch%d' % (args.lr_w, args.lr_gamma, args.max_epoch, args.decay_epoch, args.batch_size)
loss_str = 'train_D_orig_beta%s_lc%s_%s_early_bird' % (args.beta, args.lc, args.lc_layer)
args.output_dir = os.path.join('output', '%s_%s' % (loss_str, opt_str))

args.pth_dir = os.path.join(args.output_dir, 'pth')
args.img_dir = os.path.join(args.output_dir, 'img')
args.gamma_dir = os.path.join(args.output_dir, 'gamma')
args.fid_buffer_dir = os.path.join(args.output_dir, 'fid_buffer')
args.do_IS = False
args.do_FID = False
create_dir(args.pth_dir), create_dir(args.img_dir), create_dir(args.gamma_dir), create_dir(args.fid_buffer_dir)

# gpu:
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.all_masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model, percent):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def early_bird_emerge(self, model):
        mask = self.pruning(model, self.percent)
        self.put(mask)
        self.all_masks.append(mask)
        flag = self.cal_dist()
        if flag == True:
            print(self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False

best_prec1 = 0.
flag_30 = True
flag_50 = True
flag_70 = True
early_bird_30 = EarlyBird(0.3)
early_bird_50 = EarlyBird(0.5)
early_bird_70 = EarlyBird(0.7)

# set tf env
if args.do_IS:
    _init_inception()
if args.do_FID:
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

# G0:
with torch.no_grad():
    # define model
    G0 = Generator(bottom_width=args.bottom_width, gf_dim=args.gf_dim, latent_dim=args.latent_dim).cuda()
    # load ckpt
    pth_path = os.path.join('logs', 'sngan_cifar10_2020_07_04_03_54_30', 'Model', 'checkpoint_best.pth')
    checkpoint = torch.load(pth_path)
    G0.load_state_dict(checkpoint['avg_gen_state_dict'])
    best_dense_epoch = checkpoint['epoch']
    print('loaded from %s, epoch %d' % (pth_path, best_dense_epoch))
    # parallel
    # G0 = nn.DataParallel(G0)
    print('G0:', G0)
# no grad
for param in G0.parameters():
    param.requires_grad = False

# G
G = Generator(bottom_width=args.bottom_width, gf_dim=args.gf_dim, latent_dim=args.latent_dim).cuda().eval()
D = Discriminator(args=args).cuda()
# intialize G as G0:
G.load_state_dict(checkpoint['avg_gen_state_dict'])
D.load_state_dict(checkpoint['dis_state_dict'])
# parallel:
# G = nn.DataParallel(G)
print('G:', G)
print('D:', D)

# preload models:

vgg = VGGFeature(classi = True).cuda()

# param list:
W_lst, gamma_lst = [], []
for name, para in G.named_parameters():
    if 'weight' in name and para.ndimension() == 1:
        gamma_lst.append(para)
    else:
        W_lst.append(para)
print('gamma_lst:', len(gamma_lst))
for para in gamma_lst:
    print(para.size())


D_optimizer = torch.optim.Adam(D.parameters(), 0.001)
D_scheduler = torch.optim.lr_scheduler.MultiStepLR(D_optimizer, 
    milestones=[int(args.max_epoch/2), int(args.max_epoch*0.75)])
# set optimizer
W_optimizer = torch.optim.Adam(W_lst, args.lr_w)
W_scheduler = torch.optim.lr_scheduler.LambdaLR(
    W_optimizer, lr_lambda=LambdaLR(args.max_epoch, 0, args.decay_epoch).step)
gamma_optimizer = torch.optim.SGD(gamma_lst, args.lr_gamma, momentum=0.5)
gamma_scheduler = torch.optim.lr_scheduler.MultiStepLR(gamma_optimizer, 
    milestones=[int(args.max_epoch/2), int(args.max_epoch*0.75)])

# set up data_loader
dataset = datasets.ImageDataset(args)
train_loader = dataset.train

# fid stat
if args.dataset.lower() == 'cifar10':
    fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
else:
    raise NotImplementedError('no fid stat for %s' % args.dataset.lower())
assert os.path.exists(fid_stat)

# initial
fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
start_epoch = 0
best_fid = 1e4


N = len(train_loader)
print('number of batches: %d' % N)

# loop epoch:
loss_G_lst, loss_G_content_lst, loss_G_style_lst = [], [], []
for epoch in range(int(start_epoch), int(args.max_epoch)):
    start_time = time.time()
    # train mode
    G.train()
    # init loss
    loss_G_value, loss_G_content_value, loss_G_style_value = 0, 0, 0

    if early_bird_30.early_bird_emerge(G):
        print("[early_bird_30] Find EB!!!!!!!!!, epoch: "+str(epoch))
        if flag_30:
            save_checkpoint_cp({
            'epoch': epoch + 1,
            'generator': G.state_dict(),
            'discriminator': D.state_dict(),
            'W_optimizer': W_optimizer.state_dict(),
            'gamma_optimizer': gamma_optimizer.state_dict(),
            'best_fid': best_fid,
            'output_dir': args.output_dir
            }, output_dir=args.pth_dir, filename='EB-30-'+str(epoch+1))
            flag_30 = False

    if early_bird_50.early_bird_emerge(G):
        print("[early_bird_50] Find EB!!!!!!!!!, epoch: "+str(epoch))
        if flag_50:
            save_checkpoint_cp({
            'epoch': epoch + 1,
            'generator': G.state_dict(),
            'discriminator': D.state_dict(),
            'W_optimizer': W_optimizer.state_dict(),
            'gamma_optimizer': gamma_optimizer.state_dict(),
            'best_fid': best_fid,
            'output_dir': args.output_dir
            }, output_dir=args.pth_dir, filename='EB-50-'+str(epoch+1))
            flag_50 = False

    if early_bird_70.early_bird_emerge(G):
        print("[early_bird_70] Find EB!!!!!!!!!, epoch: "+str(epoch))
        if flag_70:
            save_checkpoint_cp({
            'epoch': epoch + 1,
            'generator': G.state_dict(),
            'discriminator': D.state_dict(),
            'W_optimizer': W_optimizer.state_dict(),
            'gamma_optimizer': gamma_optimizer.state_dict(),
            'best_fid': best_fid,
            'output_dir': args.output_dir
            }, output_dir=args.pth_dir, filename='EB-70-'+str(epoch+1))
            flag_70 = False
    # loop batch:
    for iter_idx, (imgs, _) in enumerate(train_loader):
        # if iter_idx > 10:
        #     break
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        W_optimizer.zero_grad()
        gamma_optimizer.zero_grad()
        D_optimizer.zero_grad()
        
        # foreward:
        imgs = imgs.cuda()
        gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim)))
        gen_imgs = G(gen_z)
        soft_gt = G0(gen_z)

        # vgg features:
        gen_imgs = nn.functional.upsample(gen_imgs, (256,256))
        soft_gt = nn.functional.upsample(soft_gt, (256,256))
        # print('gen_imgs:', gen_imgs.size())
        gen_imgs_vgg, fake_output = vgg(gen_imgs)
        soft_gt_vgg, _ = vgg(soft_gt)
        _, real_output = vgg(imgs)
        
        # content loss
        if args.lc == 'vgg':
            if args.lc_layer == 'relu1_2':
                content_loss = torch.nn.L1Loss()(gen_imgs_vgg.relu1_2, soft_gt_vgg.relu1_2)
            elif args.lc_layer == 'relu2_2':
                content_loss = torch.nn.L1Loss()(gen_imgs_vgg.relu2_2, soft_gt_vgg.relu2_2)
            elif args.lc_layer == 'relu3_3':
                content_loss = torch.nn.L1Loss()(gen_imgs_vgg.relu3_3, soft_gt_vgg.relu3_3)
        elif args.lc == 'mse':
            content_loss = torch.nn.MSELoss()(gen_imgs, soft_gt)
        else:
            raise Exception('Wrong content loss %s' % args.lc)

        # style loss
        if args.beta != 0:
            style_loss = 0
            for _, (vf_g, vf_c) in enumerate(zip(gen_imgs_vgg, soft_gt_vgg)):
                # print('vf_g:', vf_g.size())
                gm_g, gm_c = gram_matrix(vf_g), gram_matrix(vf_c)
                # print('gm_g:', gm_g.size())
                style_loss += nn.functional.mse_loss(gm_g, gm_c)

        # Total loss
        loss = content_loss
        if args.beta != 0:
            loss += args.beta * style_loss 

        # backward:
        loss.backward()

        # update:
        W_optimizer.step()
        gamma_optimizer.step()
        
        # train D
        W_optimizer.zero_grad()
        gamma_optimizer.zero_grad()
        D_optimizer.zero_grad()
        gen_imgs = G(gen_z)
        real_validity = D(imgs)
        fake_validity = D(gen_imgs)
        
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
                 
        d_loss.backward()
        D_optimizer.step()
        W_optimizer.step()
        gamma_optimizer.step()
        
        # proximal gradient for channel pruning:
        current_lr = gamma_scheduler.get_lr()[0]
        # Removed 

        loss_G_value += loss.data
        loss_G_content_value += content_loss.data
        if args.beta != 0:
            loss_G_style_value += style_loss.data * args.beta
        
        # verbose
        if iter_idx % args.print_freq == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [content loss: %f] [style loss: %f] [lr: %.4f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), 
                content_loss.item(), style_loss.item()*args.beta, current_lr) )
    
    # adjust learning rate
    W_lr = W_scheduler.step()
    gamma_lr = gamma_scheduler.step()

    # plot training loss:
    print('time: %.2f' % (time.time()-start_time))
    losses = {'loss': (loss_G_lst, loss_G_value), 
            'loss_content': (loss_G_content_lst, loss_G_content_value), 
            }
    if args.beta != 0:
        losses['loss_style'] = (loss_G_style_lst, loss_G_style_value)
    for key in losses:
        losses[key][0].append(losses[key][1]/N)
        plt.plot(losses[key][0])
        plt.savefig(os.path.join(args.output_dir, '%s.png' % key))
        plt.close()
    # evaluate:
    if (epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1) and (args.do_IS or args.do_FID):
        inception_score, fid_score = \
            calculate_metrics(args.fid_buffer_dir, args.num_eval_imgs, args.eval_batch_size, args.latent_dim, 
                            fid_stat, G, do_IS=args.do_IS, do_FID=args.do_FID)
        # save FID and IS results:
        print('Inception score: %.4f, FID score: %.4f || @ epoch %d.' % (inception_score, fid_score, epoch))
    if epoch % 5 == 0 or epoch == int(args.max_epoch)-1:
        # save generated images:
        gen_img_big = validate_cp(fixed_z, G)
        soft_gt_img_big = validate_cp(fixed_z, G0)
        imsave(os.path.join(args.img_dir, 'epoch%d_gen_img.png' % epoch), img_as_ubyte(gen_img_big) )
        if epoch == 0:
            imsave(os.path.join(args.img_dir, 'epoch%d_soft_gt_img.png' % epoch), img_as_ubyte(soft_gt_img_big) )
        # show_sparsity:
        show_sparsity(G, os.path.join(args.gamma_dir, 'gamma_%d' % epoch))
        # save pth:
        save_checkpoint_cp({
            'epoch': epoch + 1,
            'generator': G.state_dict(),
            'discriminator': D.state_dict(),
            'W_optimizer': W_optimizer.state_dict(),
            'gamma_optimizer': gamma_optimizer.state_dict(),
            'best_fid': best_fid,
            'output_dir': args.output_dir
        }, output_dir=args.pth_dir, filename='epoch%d.pth' % epoch)

    



import pickle
pickle.dump(early_bird_30.all_masks, open("mask30.pkl", "wb"))
pickle.dump(early_bird_50.all_masks, open("mask50.pkl", "wb"))
pickle.dump(early_bird_70.all_masks, open("mask70.pkl", "wb"))