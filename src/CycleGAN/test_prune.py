#!/usr/bin/python3

import argparse, sys, os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from models import Generator, Generator_ori
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataset', type=str, default='summer2winter_yosemite')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--upsample', default='transconv', choices=['ori', 'transconv', 'nearest', 'bilinear'], help='which upsample method to use in generater')
parser.add_argument('--gpu', default='3')
parser.add_argument('--evaluate-all', action="store_true")
parser.add_argument("--percent", type=float)
parser.add_argument("--extra-suffix", type=str, default=None)
# parser.add_argument('--model_id', type=int, default=199, help='indicate the model id to specify the x epoch\'s models')
opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
dataset_dir = os.path.join('datasets', opt.dataset)

###### Definition of variables ######
# Networks
with torch.no_grad():
    if opt.upsample == 'ori':
        netG_A2B = Generator_ori(opt.input_nc, opt.output_nc)
        netG_B2A = Generator_ori(opt.output_nc, opt.input_nc)
    else:
        netG_A2B = Generator(opt.input_nc, opt.output_nc)
        netG_B2A = Generator(opt.output_nc, opt.input_nc)

netG_A2B.cuda()
netG_B2A.cuda()

# Load state dicts
if opt.extra_suffix is not None:
    path_to_dir = "output_{}_{}_{}".format(opt.upsample, opt.percent, opt.extra_suffix)
else:
    path_to_dir = "output_{}_{}".format(opt.upsample, opt.percent)
    
if opt.extra_suffix is not None:
    test_output_path = os.path.join("test_set_result_{}_{}_{}".format(opt.upsample, opt.percent, opt.extra_suffix), opt.dataset)
else:
    test_output_path = os.path.join("test_set_result_{}_{}".format(opt.upsample, opt.percent), opt.dataset)

if not os.path.exists(os.path.join(test_output_path, 'A')):
    os.makedirs(os.path.join(test_output_path, 'A'))
if not os.path.exists(os.path.join(test_output_path, 'B')):
    os.makedirs(os.path.join(test_output_path, 'B'))
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
test_dataloader = DataLoader(ImageDataset(dataset_dir, transforms_=transforms_, mode='test'), 
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)


def pruning_generate(model, state_dict):

    parameters_to_prune =[]
    for (name, m) in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m = prune.custom_from_mask(m, name = 'weight', mask = state_dict[name + ".weight_mask"])
            
if not opt.evaluate_all:
    generator_A2B = os.path.join(path_to_dir, opt.dataset, 'pth', 'netG_A2B_epoch_%d.pth' % 199)
    generator_B2A = os.path.join(path_to_dir, opt.dataset, 'pth', 'netG_B2A_epoch_%d.pth' % 199)
    #generator_A2B  = "output/netG_A2B.pth"
    #generator_B2A  = "output/netG_B2A.pth"
    a2b_dict = torch.load(generator_A2B)
    b2a_dict = torch.load(generator_B2A)
    pruning_generate(netG_A2B, a2b_dict)
    pruning_generate(netG_B2A, b2a_dict)
    netG_A2B.load_state_dict(a2b_dict)
    netG_B2A.load_state_dict(b2a_dict)

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    

    for i, batch in enumerate(test_dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

        # Save image files
        save_image( fake_A, os.path.join(test_output_path, 'A/%04d.png' % (i+1)) )
        save_image( fake_B, os.path.join(test_output_path, 'B/%04d.png' % (i+1)) )

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(test_dataloader)))

    sys.stdout.write('\n')
    ###################################
else:
    for epoch in range(0, 200, 10):
        if not os.path.exists(os.path.join(test_output_path, str(epoch))):
            os.makedirs(os.path.join(test_output_path, str(epoch)))
        if not os.path.exists(os.path.join(test_output_path, str(epoch))):
            os.makedirs(os.path.join(test_output_path, str(epoch)))
        
        if not os.path.exists(os.path.join(test_output_path, str(epoch), 'A')):
            os.makedirs(os.path.join(test_output_path, str(epoch), 'A'))
        if not os.path.exists(os.path.join(test_output_path, str(epoch), 'B')):
            os.makedirs(os.path.join(test_output_path, str(epoch), 'B'))
        generator_A2B = os.path.join(path_to_dir, opt.dataset, 'pth', 'netG_A2B_epoch_%d.pth' % epoch)
        generator_B2A = os.path.join(path_to_dir, opt.dataset, 'pth', 'netG_B2A_epoch_%d.pth' % epoch)
        #generator_A2B  = "output/netG_A2B.pth"
        #generator_B2A  = "output/netG_B2A.pth"
        a2b_dict = torch.load(generator_A2B)
        b2a_dict = torch.load(generator_B2A)
        pruning_generate(netG_A2B, a2b_dict)
        pruning_generate(netG_B2A, b2a_dict)
        netG_A2B.load_state_dict(a2b_dict)
        netG_B2A.load_state_dict(b2a_dict)

        # Set model's test mode
        netG_A2B.eval()
        netG_B2A.eval()

        for i, batch in enumerate(test_dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            # Generate output
            fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
            fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

            # Save image files
            save_image( fake_A, os.path.join(test_output_path, str(epoch), 'A/%04d.png' % (i+1)) )
            save_image( fake_B, os.path.join(test_output_path, str(epoch), 'B/%04d.png' % (i+1)) )

            sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(test_dataloader)))

        sys.stdout.write('\n')
        
    i = 199
    if not os.path.exists(os.path.join(test_output_path, '199')):
        os.makedirs(os.path.join(test_output_path, '199'))
    if not os.path.exists(os.path.join(test_output_path, '199')):
        os.makedirs(os.path.join(test_output_path, '199'))
        
    if not os.path.exists(os.path.join(test_output_path, '199', 'A')):
        os.makedirs(os.path.join(test_output_path, '199', 'A'))
    if not os.path.exists(os.path.join(test_output_path, '199', 'B')):
        os.makedirs(os.path.join(test_output_path, '199', 'B'))
    generator_A2B = os.path.join(path_to_dir, opt.dataset, 'pth', 'netG_A2B_epoch_%d.pth' % i)
    generator_B2A = os.path.join(path_to_dir, opt.dataset, 'pth', 'netG_B2A_epoch_%d.pth' % i)
    #generator_A2B  = "output/netG_A2B.pth"
    #generator_B2A  = "output/netG_B2A.pth"
    a2b_dict = torch.load(generator_A2B)
    b2a_dict = torch.load(generator_B2A)
    pruning_generate(netG_A2B, a2b_dict)
    pruning_generate(netG_B2A, b2a_dict)
    netG_A2B.load_state_dict(a2b_dict)
    netG_B2A.load_state_dict(b2a_dict)

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    for i, batch in enumerate(test_dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

        # Save image files
        save_image( fake_A, os.path.join(test_output_path, '199/A/%04d.png' % (i+1)) )
        save_image( fake_B, os.path.join(test_output_path, '199/B/%04d.png' % (i+1)) )

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(test_dataloader)))

    sys.stdout.write('\n')