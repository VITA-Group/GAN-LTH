'''
Deprecated. Need tune parameter for interpolation + conv

Based on original CycleGAN model defined in models.py
Modifications are made:
1. Tansconv -> upsample + conv
2. Use dim_lst to define channel numbers.
'''

import torch.nn as nn
import torch.nn.functional as F

class InterpolateLayer(nn.Module):
    ''' Nearest pixel interplotation. Scale by 2x2
    '''
    def __init__(self):
        super(InterpolateLayer, self).__init__()
    def __call__(self, x):
        return nn.functional.interpolate(x, scale_factor=(2,2), mode='nearest')


class ResidualBlock(nn.Module):
    def __init__(self, in_features=256, mid_features=256):
        super(ResidualBlock, self).__init__()
        
        if mid_features > 0:
            conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, mid_features, 3),
                        nn.InstanceNorm2d(mid_features, affine=True),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(mid_features, in_features, 3),
                        nn.InstanceNorm2d(in_features, affine=False)  ]

            self.conv_block = nn.Sequential(*conv_block)
        else:
            self.conv_block = nn.Sequential()

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, dim_lst=None, upsample='nearest'):
        super(Generator, self).__init__()

        if dim_lst is None:
            dim_lst = [64, 128] + [256]*n_residual_blocks + [128, 64]
        assert len(dim_lst) == 4 + n_residual_blocks

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, dim_lst[0], 7),
                    nn.InstanceNorm2d(dim_lst[0], affine=True),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        model += [  nn.Conv2d(dim_lst[0], dim_lst[1], 3, stride=2, padding=1),
                    nn.InstanceNorm2d(dim_lst[1], affine=True),
                    nn.ReLU(inplace=True) ]
        
        model += [  nn.Conv2d(dim_lst[1], 256, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(256, affine=False),
                    nn.ReLU(inplace=False) ]

        # Residual blocks
        for i in range(n_residual_blocks):
            model += [ResidualBlock(in_features=256, mid_features=dim_lst[2+i])]

        # Upsampling
        if upsample == 'transconv':
            print('+++ Using transconv +++')
            model += [  nn.ConvTranspose2d(256, dim_lst[2+n_residual_blocks], 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(dim_lst[2+n_residual_blocks], affine=True),
                        nn.ReLU(inplace=True) ]
            model += [  nn.ConvTranspose2d(dim_lst[2+n_residual_blocks], dim_lst[2+n_residual_blocks+1], 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(dim_lst[2+n_residual_blocks+1], affine=True),
                        nn.ReLU(inplace=True) ]
        elif upsample == 'nearest':
            print('+++ Using %s interpolate +++' % upsample)
            model += [  
                InterpolateLayer(),
                nn.Conv2d(256, dim_lst[2+n_residual_blocks], 3, stride=1, padding=1),
                nn.InstanceNorm2d(dim_lst[2+n_residual_blocks], affine=True),
                nn.ReLU(inplace=True) 
            ]
            model += [ 
                InterpolateLayer(), 
                nn.Conv2d(dim_lst[2+n_residual_blocks], dim_lst[2+n_residual_blocks+1], 3, stride=1, padding=1),
                nn.InstanceNorm2d(dim_lst[2+n_residual_blocks+1], affine=True),
                nn.ReLU(inplace=True) 
            ]
        elif upsample == 'bilinear':
            print('+++ Using %s interpolate +++' % upsample)
            model += [  
                nn.Upsample(scale_factor = 2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, dim_lst[2+n_residual_blocks], 3, stride=1, padding=0),
                nn.InstanceNorm2d(dim_lst[2+n_residual_blocks], affine=True),
                nn.ReLU(inplace=True) 
            ]
            model += [ 
                nn.Upsample(scale_factor = 2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim_lst[2+n_residual_blocks], dim_lst[2+n_residual_blocks+1], 3, stride=1, padding=0),
                nn.InstanceNorm2d(dim_lst[2+n_residual_blocks+1], affine=True),
                nn.ReLU(inplace=True) 
            ]
        else:
            raise Exception('Wrong upsample method: %s' % upsample)

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(dim_lst[2+n_residual_blocks+1], output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128, affine=True), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256, affine=True), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512, affine=True), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])


# if __name__ == "__main__":
#     from utils import measure_model, model_size
#     g = Generator(3, 3, transconv='bilinear')
#     measure_model(g, 256, 256)
#     print(model_size(g))

