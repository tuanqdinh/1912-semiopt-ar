"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import os, sys, time
import numpy as np
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils


from config import args
from utils.helper import Helper
from utils.provider import Provider
from utils.plotter import Plotter
from utils.sfc import SFC

# CONSTANTS ######################3
# name, space dim, number of points

model_name=args.model_name + '-' + args.dataset_name
dataset_path=os.path.join(args.data_path, args.dataset_name)
output_path=os.path.join(args.result_path, model_name)

# saved checkpoint
model_path=os.path.join(output_path, 'snapshots')
net_path=os.path.join(model_path, 'net.pth')
sample_path=os.path.join(output_path, 'samples')
log_path=os.path.join(output_path, "log.txt")
writer_path=os.path.join(output_path, 'runs')
# makedir
Helper.mkdir(args.result_path)
Helper.mkdir(output_path)
Helper.mkdir(model_path)
Helper.mkdir(sample_path)
logf=open(log_path, 'w')
Helper.mkdir(writer_path, rm=True)
writer=SummaryWriter(comment = model_name, log_dir = writer_path)

#####====================== Data ================######
device=Helper.__device__

####====== Modules =======####
def log_loss(epoch, step, total_step, loss, start_time):
	# convert
	loss=loss.cpu().data.numpy()
	# msg
	message='Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time: {:.4f}s'.format(
		epoch, args.num_epochs, step, total_step, loss, time.time() - start_time)
	# log out
	Helper.log(logf, message)

def log_loss2(epoch, step, total_step, D_cost, G_cost, start_time):
	# convert
	loss_d = D_cost.cpu().data.numpy()
	loss_g = G_cost.cpu().data.numpy()
	# msg
	message = 'Epoch [{}/{}], Step [{}/{}], LossD: {:.4f}, LossG: {:.4f}, time: {:.4f}s'.format(epoch, args.num_epochs, step, total_step, loss_d, loss_g, time.time() - start_time)
	# log out
	Helper.log(logf, message)

def plot_step(sample):
	# reshape [C, H, W] -> [W, H, C]
	# sample = sample.permute(2, 1, 0)
	# sample = sample.contiguous().view(1024, 3)
	# sample = sample.cpu().data.numpy()
	# plot
	for i in range(1024):
		img = sample[:i+1, :]
		filepath = sample_path + '/step_{:04d}.png'.format(i)
		Plotter.plot_pc_color(img, filepath)
		print(i)

def plot_samples(samples, epoch, name, nsamples=1, color=False):
	# reshape [N, C, H, W] -> [N, W, H, C]
	samples = samples.permute(0, 3, 2, 1)
	samples = samples.contiguous().view(samples.shape[0], 1024, 3)
	samples = samples.cpu().data.numpy()
	# plot
	for i in range(nsamples):
		img = samples[i, :, :]
		filepath = sample_path + '/{}_{}_{}.png'.format(name, epoch, i)
		if color:
			Plotter.plot_pc_color(img, filepath)
		else:
			Plotter.plot_pc(img, filepath)
