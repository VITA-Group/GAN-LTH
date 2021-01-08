
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import namedtuple

class CLF(nn.Module):
	def __init__(self, norm_layer='IN'):
		super(CLF, self).__init__()

		if norm_layer=='BN':
			norm2d = nn.BatchNorm2d
		else:
			norm2d = nn.InstanceNorm2d

		# A bunch of convolutions one after another
		self.block1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, stride=2, padding=1),
			norm2d(64, affine=True), 
			nn.LeakyReLU(0.2, inplace=True) 
		)

		self.block2 = nn.Sequential(
			nn.Conv2d(64, 128, 3, stride=2, padding=1),
			norm2d(128, affine=True), 
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.block3 = nn.Sequential(
			nn.Conv2d(128, 256, 3, stride=2, padding=1),
			norm2d(256, affine=True), 
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.block4 = nn.Sequential(
			nn.Conv2d(256, 512, 3, stride=2, padding=1),
			norm2d(512, affine=True), 
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.block5 = nn.Sequential(
			nn.Conv2d(512, 16, 3, stride=2, padding=1),
			norm2d(16, affine=True), 
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.block6 = nn.Sequential(
			nn.Flatten(),
			nn.Linear(16*8*8, 2),
		)

	def forward(self, x):
		h1 = self.block1(x)
		h2 = self.block2(h1)
		h3 = self.block3(h2)
		h4 = self.block4(h3) # 32 16 16 = 2048
		h5 = self.block5(h4) # 16 8 8 = 1024
		logits = self.block6(h5)

		r = namedtuple("Outputs", ['h1', 'h2', 'h3', 'h4', 'h5', 'logits'])(h1, h2, h3, h4, h5, logits)

		return r

if __name__ == '__main__':
	clf = CLF()
	x = torch.Tensor(np.random.randn(16,3,256,256))
	f = clf(x)

	print('h1:', f.h1.size())
	print('h2:', f.h2.size())
	print('h3:', f.h3.size())
	print('h4:', f.h4.size())
	print('h5:', f.h5.size())
	print('logits:', f.logits.size())
