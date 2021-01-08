'''
from https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
'''

from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, classi = False, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.classi = classi
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size = (7,7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(4096, 2)
        )
    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        if self.classi:
            classi = self.classifier(self.avgpool(h).view(-1, 25088))
            return out, classi
        else:
            return out

if __name__ == '__main__':
    model = models.vgg16(pretrained=True)
    
    vgg_pretrained_features = model.features
    for i, feature in enumerate(vgg_pretrained_features):
        print(i, feature)