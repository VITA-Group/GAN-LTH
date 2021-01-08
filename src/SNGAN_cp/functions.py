from vgg import Vgg16
import torch
import torch.nn as nn
import torchvision
from skimage.io import imsave
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.inception_score import _init_inception, get_inception_score
from utils.fid_score import create_inception_graph, check_or_download_inception, calculate_fid_given_paths
from copy import deepcopy

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging

from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
logger = logging.getLogger(__name__)
class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


class LambdaLR():
	def __init__(self, n_epochs, offset, decay_start_epoch):
		assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
		self.n_epochs = n_epochs
		self.offset = offset
		self.decay_start_epoch = decay_start_epoch

	def step(self, epoch):
		return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class VGGFeature(nn.Module):
	def __init__(self, classi = False):
		super(VGGFeature, self).__init__()
		self.add_module('vgg', Vgg16(classi = classi))
	def __call__(self,x):
		x = (x.clone()+1.)/2. # [-1,1] -> [0,1]
		x_vgg = self.vgg(x)
		return x_vgg

def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram

def soft_sign(w, th):
	'''
	pytorch soft-sign function
	'''
	with torch.no_grad():
		temp = torch.abs(w) - th
		# print('th:', th)
		# print('temp:', temp.size())
		return torch.sign(w) * nn.functional.relu(temp)  

def validate_cp(fixed_z, G, n_row=5):
	# eval mode
	G = G.eval()

	# generate images
	gen_imgs = G(fixed_z)
	gen_img_big = fourD2threeD( np.moveaxis(gen_imgs.detach().cpu().numpy(), 1, -1), n_row=n_row )

	gen_img_big = (gen_img_big + 1)/2 # [-1,1] -> [0,1]

	return gen_img_big


def calculate_metrics(fid_buffer_dir, num_eval_imgs, eval_batch_size, latent_dim, fid_stat, G, do_IS=False, do_FID=True):
	# eval mode
	G = G.eval()

	# get fid and inception score
	if do_IS and do_FID:
		if not os.path.isdir(fid_buffer_dir):
			os.mkdir(fid_buffer_dir)

		eval_iter = num_eval_imgs // eval_batch_size
		img_list = list()
		for iter_idx in range(eval_iter):
			z = torch.cuda.FloatTensor(np.random.normal(0, 1, (eval_batch_size, latent_dim)))

			# Generate a batch of images
			gen_imgs = G(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
			for img_idx, img in enumerate(gen_imgs):
				file_name = os.path.join(fid_buffer_dir, 'iter%d_b%d.png' % (iter_idx, img_idx))
				imsave(file_name, img)
			img_list.extend(list(gen_imgs))

	# get inception score
	if do_IS:
		print('=> calculate inception score')
		mean, std = get_inception_score(img_list)
	else:
		mean, std = 0, 0

	# get fid score
	if do_FID:
		print('=> calculate fid score')
		fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
	else:
		fid_score = 0

	if do_IS and do_FID:
		os.system('rm -r {}'.format(fid_buffer_dir))

	return mean, fid_score


def show_sparsity(model, save_name, model_path=None):
	# load model if necessary:
	if model_path is not None:
		if not os.path.exists(model_path):
			raise Exception("G model path doesn't exist at %s!" % model_path)
		print('Loading generator from %s' % model_path)
		model.load_state_dict(torch.load(model_path))
	
	# get all scaler parameters form the network:
	scaler_list = []
	for m in model.modules():
		if isinstance(m, torch.nn.BatchNorm2d) and m.weight is not None:
			m_cpu = m.weight.data.cpu().numpy().squeeze()
			# print('m_cpu:', type(m_cpu), m_cpu.shape)
			scaler_list.append(m_cpu)
	all_scaler = np.concatenate(scaler_list, axis=0)
	print('all_scaler:', all_scaler.shape, 'L0 (sum):', np.sum(all_scaler!=0), 'L1 (mean):', np.mean(np.abs(all_scaler)))

	# save npy and plt png:
	# np.save(save_name + '.npy', all_scaler)
	n, bins, patches = plt.hist(all_scaler, 50)
	# print(n)
	plt.savefig(save_name + '.png')
	plt.close()

	return all_scaler


def create_dir(_path):
	if not os.path.exists(_path):
		os.makedirs(_path)


def fourD2threeD(batch, n_row=10):
	'''
	Convert a batch of images (N,W,H,C) to a single big image (W*n, H*m, C)
	Input:
		batch: type=ndarray, shape=(N,W,H,C)
	Return:
		rows: type=ndarray, shape=(W*n, H*m, C)
	'''
	N = batch.shape[0]
	img_list = np.split(batch, N)
	for i, img in enumerate(img_list):
		img_list[i] = img.squeeze(axis=0)
	one_row = np.concatenate(img_list, axis=1)
	# print('one_row:', one_row.shape)
	row_list = np.split(one_row, n_row, axis=1)
	rows = np.concatenate(row_list, axis=0)
	return rows

def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None, fix_G = False):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []
    if fix_G:
        gen_net.eval()
    else:
        gen_net.train()
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        
        tp = torch.sum(real_validity > 0)
        tn = torch.sum(fake_validity < 0)
        fn = torch.sum(real_validity <= 0)
        fp = torch.sum(fake_validity >= 0)
        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        accuracy = (tp + tn) / (tp + fn + fp + fn)
        
        fps.append(fp.item())
        tps.append(tp.item())
        fns.append(fn.item())
        tns.append(tn.item())

        writer.add_scalar('precision', precision.item(), global_steps)
        writer.add_scalar('recall', recall.item(), global_steps)
        writer.add_scalar('accuracy', accuracy.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            if not fix_G:
                g_loss.backward()
                gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
    
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)
    
def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, 'iter%d_b%d.png' % (iter_idx, img_idx))
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    os.system('rm -r {}'.format(fid_buffer_dir))

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score

def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)