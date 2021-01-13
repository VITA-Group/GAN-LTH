#!/usr/bin/python3

import argparse, itertools, os
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from models import Generator, Discriminator, Generator_ori
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataset', type=str, default='summer2winter_yosemite')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpu', default='0')
parser.add_argument('--rand', type=str)
parser.add_argument('--upsample', default='transconv', choices=['ori', 'transconv', 'nearest', 'bilinear'], help='which upsample method to use in generater')
opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
torch.backends.cudnn.benchmark = True

## mkdir:
dataset_dir = os.path.join('datasets', opt.dataset)
output_dir = os.path.join('output_%s' % opt.upsample, opt.dataset)
img_dir = os.path.join(output_dir, 'imgs')
pth_dir = os.path.join(output_dir, 'pth')
if not os.path.isdir(img_dir):
    os.makedirs(img_dir)
if not os.path.isdir(pth_dir):
    os.makedirs(pth_dir)

###### Definition of variables ######
# Networks
if opt.upsample == 'ori':
    netG_A2B = Generator_ori(opt.input_nc, opt.output_nc)
    netG_B2A = Generator_ori(opt.output_nc, opt.input_nc)
else:
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)


netG_A2B.load_state_dict(torch.load(os.path.join(opt.rand, 'netG_A2B_seed_{}.pth.tar'.format(opt.seed))))
netG_B2A.load_state_dict(torch.load(os.path.join(opt.rand, 'netG_B2A_seed_{}.pth.tar'.format(opt.seed))))
netD_A.load_state_dict(torch.load(os.path.join(opt.rand, 'netD_A_seed_{}.pth.tar'.format(opt.seed))))
netD_B.load_state_dict(torch.load(os.path.join(opt.rand, 'netD_B_seed_{}.pth.tar'.format(opt.seed))))

netG_A2B.train()
netD_A.train()
netD_B.train()
netG_B2A.train()

netG_A2B.cuda()
netG_B2A.cuda()
netD_A.cuda()
netD_B.cuda()

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(dataset_dir, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

# Loss plot
# logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
N = len(dataloader)
print('N:', N) # 1334
loss_G_lst, loss_D_lst, loss_G_GAN_lst, loss_G_cycle_lst, loss_G_identity_lst = [], [], [], [], []
for epoch in range(opt.epoch, opt.n_epochs):
    # reset loss to 0
    loss_G_value, loss_D_value, loss_G_GAN_value, loss_G_cycle_value, loss_G_identity_value = 0, 0, 0, 0, 0
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss: not in the paper, but helps training stabability.
        # See https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md#the-color-gets-inverted-from-the-beginning-of-training-249
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        
        # print loss:
        if i % 100 == 0:
            print('epoch %d-%d: loss_G %.4f, loss_D %.4f' % (
                epoch, i, loss_G.data, (loss_D_A + loss_D_B).data))

        loss_G_value += loss_G.data
        loss_D_value += (loss_D_A + loss_D_B).data
        loss_G_GAN_value += (loss_GAN_A2B + loss_GAN_B2A).data
        loss_G_cycle_value += (loss_cycle_ABA + loss_cycle_BAB).data
        loss_G_identity_value += (loss_identity_A + loss_identity_B).data
    if epoch % 5 == 0 or epoch == opt.n_epochs - 1:
        torch.save(netG_A2B.state_dict(), os.path.join(pth_dir, 'netG_A2B_epoch_{}.pth'.format(epoch)))
        torch.save(netG_B2A.state_dict(), os.path.join(pth_dir, 'netG_B2A_epoch_{}.pth'.format(epoch)))
        torch.save(netD_A.state_dict(), os.path.join(pth_dir, 'netD_A_epoch_{}.pth'.format(epoch)))
        torch.save(netD_B.state_dict(), os.path.join(pth_dir, 'netD_B_epoch_{}.pth'.format(epoch)))
    ## at the end of each epoch
    # plot loss:
    losses = {'loss_G': (loss_G_lst, loss_G_value), 'loss_D': (loss_D_lst, loss_D_value), 
            'loss_G_GAN': (loss_G_GAN_lst, loss_G_GAN_value), 
            'loss_G_cycle': (loss_G_cycle_lst, loss_G_cycle_value), 
            'loss_G_identity': (loss_G_identity_lst, loss_G_identity_value)}
    for key in losses:
        losses[key][0].append(losses[key][1]/N)
        plt.plot(losses[key][0])
        plt.savefig(os.path.join(output_dir, '%s.png' % key))
        plt.close()
    # save imgs:
    if epoch % 5 == 0 or epoch == opt.n_epochs - 1:
        images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B}
        for key in images:
            img_np = images[key].detach().cpu().numpy().squeeze()
            img_np = (img_np + 1) / 2 # (-1,1) -> (0,1)
            img_np = img_as_ubyte(np.moveaxis(img_np, 0, -1)) # channel first to channel last.
            print('img_np:', img_np.shape)
            imsave(os.path.join(output_dir, 'epoch%d_%s.png' % (epoch, key)), img_np)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
