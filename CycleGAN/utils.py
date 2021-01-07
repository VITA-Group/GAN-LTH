import random
import time
import datetime
import sys
import os

from torch.autograd import Variable
import torch
import numpy as np

from datasets import ImageDataset
from fid_score import calculate_fid_given_paths
from torchvision.utils import save_image

from datetime import datetime
import dateutil.tz

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)      
       

class ReplayBuffer():
    '''
    follow Shrivastava et al.’s strategy: 
    update D using a history of generated images, rather than the ones produced by the latest generators. 
    '''
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def validate(test_dataloader, netG_A2B, netG_B2A, dataset):
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    temp_buffer = "fid_buffer_{}".format(timestamp)
    if not os.path.exists(temp_buffer):
        os.mkdir(temp_buffer)
        os.mkdir(os.path.join(temp_buffer, 'A'))
        os.mkdir(os.path.join(temp_buffer, 'B'))
        
    for i, batch in enumerate(test_dataloader):
        # Set model input
        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()

        # Generate output
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

        # Save image files
        
        save_image( fake_A, os.path.join(temp_buffer, 'A', '%04d.png' % (i+1)))
        save_image( fake_A, os.path.join(temp_buffer, 'B', '%04d.png' % (i+1)))
        
    fid_value_A = calculate_fid_given_paths(['datasets/{}/train/A'.format(dataset), os.path.join(temp_buffer, 'A')] ,
                                          1,
                                          True,
                                          2048)
        
    fid_value_B = calculate_fid_given_paths(['datasets/{}/train/B'.format(dataset), os.path.join(temp_buffer, 'B')] ,
                                          1,
                                          True,
                                          2048)
        
    os.system('rm -rf {}'.format(temp_buffer))
    print("FID A: {}, FID B: {}".format(fid_value_A, fid_value_B))
    return fid_value_A, fid_value_B