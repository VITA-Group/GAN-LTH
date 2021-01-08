'''
Useless
Transfer training set to target styles.
B_soft <- G1(A)
A_soft <- G2(B)
'''

import argparse, sys, os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets_mine_better import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataset', type=str, default='summer2winter_yosemite')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpu', default='3')
opt = parser.parse_args()
opt.generator_A2B = os.path.join('output_transconv', opt.dataset, 'pth', 'netG_A2B_epoch_%d.pth' % 199)
opt.generator_B2A = os.path.join('output_transconv', opt.dataset, 'pth', 'netG_B2A_epoch_%d.pth' % 199)
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
dataset_dir = os.path.join('datasets', opt.dataset)

###### Definition of variables ######
# Networks
with torch.no_grad():
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)

netG_A2B.cuda()
netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader_A_train = DataLoader(
    ImageDataset(os.path.join('datasets', opt.dataset, 'train/A'), transforms_=transforms_), 
    batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
dataloader_B_train = DataLoader(
    ImageDataset(os.path.join('datasets', opt.dataset, 'train/B'), transforms_=transforms_), 
    batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
soft_training_set = os.path.join('train_set_result', opt.dataset)
if not os.path.exists(os.path.join(soft_training_set, 'A')):
    os.makedirs(os.path.join(soft_training_set, 'A'))
if not os.path.exists(os.path.join(soft_training_set, 'B')):
    os.makedirs(os.path.join(soft_training_set, 'B'))

for i, batch in enumerate(dataloader_A_train):
    real_A = Variable(input_A.copy_(batch))
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    save_image( fake_B, os.path.join(soft_training_set, 'B/%04d.png' % (i+1)) )

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader_A_train)))

for i, batch in enumerate(dataloader_B_train):
    real_B = Variable(input_B.copy_(batch))
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
    save_image( fake_A, os.path.join(soft_training_set, 'A/%04d.png' % (i+1)) )

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader_B_train)))

sys.stdout.write('\n')
###################################
