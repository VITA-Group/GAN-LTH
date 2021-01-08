import argparse, itertools, os, time
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F 
from PIL import Image
from skimage.io import imsave
from skimage import img_as_ubyte
import torch
import torch.nn as nn

from models.models import Generator, Discriminator, Discriminator2
from utils.utils import *
from utils.perceptual import *

def mask_to_rgb(mask):
    '''
    Given the Cityscapes mask file, this converts the ids into rgb colors.
    This is needed as we are interested in a sub-set of labels, thus can't just use the
    standard color output provided by the dataset.
    '''

    mappingrgb = {
            0: (255, 0, 0),  # unlabeled
            1: (255, 0, 0),  # ego vehicle
            2: (255, 0, 0),  # rect border
            3: (255, 0, 0),  # out of roi
            4: (255, 0, 0),  # static
            5: (255, 0, 0),  # dynamic
            6: (255, 0, 0),  # ground
            7: (0, 255, 0),  # road
            8: (255, 0, 0),  # sidewalk
            9: (255, 0, 0),  # parking
            10: (255, 0, 0),  # rail track
            11: (255, 0, 0),  # building
            12: (255, 0, 0),  # wall
            13: (255, 0, 0),  # fence
            14: (255, 0, 0),  # guard rail
            15: (255, 0, 0),  # bridge
            16: (255, 0, 0),  # tunnel
            17: (255, 0, 0),  # pole
            18: (255, 0, 0),  # polegroup
            19: (255, 0, 0),  # traffic light
            20: (255, 0, 0),  # traffic sign
            21: (255, 0, 0),  # vegetation
            22: (255, 0, 0),  # terrain
            23: (0, 0, 255),  # sky
            24: (255, 0, 0),  # person
            25: (255, 0, 0),  # rider
            26: (255, 255, 0),  # car
            27: (255, 0, 0),  # truck
            28: (255, 0, 0),  # bus
            29: (255, 0, 0),  # caravan
            30: (255, 0, 0),  # trailer
            31: (255, 0, 0),  # train
            32: (255, 0, 0),  # motorcycle
            33: (255, 0, 0),  # bicycle
            -1: (255, 0, 0)  # licenseplate
        }
    
    rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
    for k in mappingrgb:
        rgbimg[0][mask == k] = mappingrgb[k][0]
        rgbimg[1][mask == k] = mappingrgb[k][1]
        rgbimg[2][mask == k] = mappingrgb[k][2]
    return rgbimg


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='3')
parser.add_argument('--cpus', default=4)
parser.add_argument('--quant', action='store_true', help='enable quantization (for both activation and weight)')
args = parser.parse_args()


model_str = 'GS8v2_rho0.002_beta0.001_vgg_e200-b8_sgd_mom0.5_lrgamma0.1_adam_lrw1e-05_wd0.001'
gen_photo_root_dir = os.path.join(model_str, 'gen')

real_photo_root_dir = os.path.join('/home/haotao/PyTorch-CycleGAN/datasets/leftImg8bit/val')
label_root_dir = os.path.join('/home/haotao/PyTorch-CycleGAN/datasets/gtFine/val')

# network:
netG = Generator(3, 3, quant=args.quant).cuda()
g_path = os.path.join('results/cityscapes/B2A', model_str, 'pth/epoch199_netG.pth')
netG.load_state_dict(torch.load(g_path))

# generate photo from labels:
with torch.no_grad():
    for city in os.listdir(real_photo_root_dir):
        real_photo_dir = os.path.join(real_photo_root_dir, city)
        print('real_photo_dir:', real_photo_dir)
        for real_photo_file_name in os.listdir(real_photo_dir):
            # print('realphoto_file_name:', real_photo_file_name)
            seg_color_file_name = '{}_{}'.format(real_photo_file_name.split('_leftImg8bit')[0], 'gtFine_color.png')
            # print('seg_color_file_name:', seg_color_file_name)
            gen_photo_file_name = real_photo_file_name
            # print('gen_photo_file_name:', gen_photo_file_name)
            seg_color = Image.open(os.path.join(label_root_dir, city, seg_color_file_name)).convert('RGB').resize((256, 256))
            # print('seg_color:', seg_color.size, seg_color.getextrema())
            seg_color = np.array(seg_color, dtype=np.uint8)
            seg_color = np.moveaxis(seg_color, -1, 0)
            seg_color = torch.from_numpy(seg_color).float().cuda()
            seg_color = torch.unsqueeze(seg_color, dim=0)
            # print('seg_color:', seg_color.size(), torch.max(seg_color), torch.min(seg_color)) # range [0,255]
            seg_color /= 255
            seg_color = seg_color * 2 - 1 # range [-1,1]

            # generate photos:
            photo_gen = netG(seg_color)
            # print('photo_gen:', photo_gen.size()) # size=(3,H,W)

            # save figures:
            photo_gen = photo_gen.data.cpu().numpy()
            photo_gen = np.squeeze(photo_gen, axis=0)
            photo_gen = np.moveaxis(photo_gen, 0, -1)
            # print('photo_gen:', photo_gen.shape, np.amax(photo_gen), np.amin(photo_gen))
            photo_gen = (photo_gen + 1)/2
            # print('photo_gen:', photo_gen.shape, np.amax(photo_gen), np.amin(photo_gen))
            gen_photo_dir = os.path.join('results/cityscapes/B2A', gen_photo_root_dir, city)
            create_dir(gen_photo_dir)
            imsave(os.path.join(gen_photo_dir, gen_photo_file_name), img_as_ubyte(photo_gen))

            
            # break

        


