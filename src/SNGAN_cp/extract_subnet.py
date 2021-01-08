import torch
import torch.nn as nn

import argparse, os
import numpy as np 
from models.sngan_cifar10 import Generator

from utils.utils import measure_model, model_param_num

def extract_subnet(state_dict):

	# import network
	G = Generator(bottom_width=args.bottom_width, gf_dim=args.gf_dim, latent_dim=args.latent_dim)
	G.load_state_dict(state_dict)

	# measure dense model
	# measure_model(G, z_dim=args.latent_dim)
	print(model_param_num(G))

	# extract:
	dim_lst, selected_input_channel_idx_lst = [], []
	for name, m in G.named_modules():
		if isinstance(m, nn.BatchNorm2d) and m.weight is not None:
			gamma = m.weight.data.detach().cpu().numpy()
			none_zero_dim = np.sum(gamma!=0)
			dim_lst.append(none_zero_dim)
			selected_idx = np.where(gamma!=0)[0]
			selected_input_channel_idx_lst.append(selected_idx)
			# print(name)
	print('dim_lst:', dim_lst, len(dim_lst))
	print('selected_input_channel_idx_lst:', len(selected_input_channel_idx_lst))
	assert len(dim_lst) == 7 # for cartoongan generator

	# get hidden_dim_lst:
	hidden_dim_lst = dim_lst[1:6:2] # 1,3,5

	# selected_input_channel_idx_lst:
	selected_input_channel_idx_lst = selected_input_channel_idx_lst[0::2] # 0,2,4,6

	# define subnet:
	G_sub = Generator(bottom_width=args.bottom_width, gf_dim=args.gf_dim, latent_dim=args.latent_dim,
			hidden_dim_lst=hidden_dim_lst, selected_input_channel_idx_lst=selected_input_channel_idx_lst)


	# measure sub model
	measure_model(G_sub, z_dim=args.latent_dim)
	print(model_param_num(G_sub))


if __name__ == '__main__':
	# args:
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default='7', help='which gpu to use')

	parser.add_argument('--gf_dim', type=int, default=256, help='The base channel num of gen')
	parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
	parser.add_argument('--bottom_width', type=int, default=4, help="the base resolution of the GAN")
	parser.add_argument('--dir', type=str)
	parser.add_argument('--load-epoch', type=int, default=99)

	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	exp_str = args.dir
	args.load_path = os.path.join( 
		exp_str,
		'pth', 'epoch{}.pth'.format(args.load_epoch))
	print("=> Loading checkpoint from {}".format(args.load_path))
	# state dict:
 
	assert os.path.exists(args.load_path)
	checkpoint = torch.load(args.load_path)
	print('=> loaded checkpoint %s' % args.load_path)
	state_dict = checkpoint['generator']

	#
	extract_subnet(state_dict)
