#!/usr/bin/python3
import random

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
parser.add_argument('--seed', type=int, default = None)
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--upsample', default='transconv', choices=['ori', 'transconv', 'nearest', 'bilinear'], help='which upsample method to use in generater')
opt = parser.parse_args()
print(opt)

random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)

# Networks
if opt.upsample == 'ori':
    netG_A2B = Generator_ori(opt.input_nc, opt.output_nc)
    netG_B2A = Generator_ori(opt.output_nc, opt.input_nc)
else:
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

netG_A2B.cuda()
netG_B2A.cuda()
netD_A.cuda()
netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

torch.save(netG_A2B.state_dict(), "initial_weights/netG_A2B_seed_{}.pth.tar".format(opt.seed))
torch.save(netG_B2A.state_dict(), "initial_weights/netG_B2A_seed_{}.pth.tar".format(opt.seed))
torch.save(netD_A.state_dict(), "initial_weights/netD_A_seed_{}.pth.tar".format(opt.seed))
torch.save(netD_B.state_dict(), "initial_weights/netD_B_seed_{}.pth.tar".format(opt.seed))




