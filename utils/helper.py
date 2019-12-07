"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import os
import shutil
import torch
import numpy as np
import pymorton as pm


class Helper:

	__device__ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	############################################
	@staticmethod
	def deactivate(model):
		for p in model.parameters():
			p.requires_grad = False

	@staticmethod
	def activate(model):
		for p in model.parameters():
			p.requires_grad = True
	################### Nets ##################

	@staticmethod
	def sample_z(m, n):
		return np.random.normal(size=[m, n], loc=0, scale=1)
		# return np.random.uniform(-1., 1., size=[m, n])

	@staticmethod
	def save_sample(samples, sample_path, label):
		# save npy
		filepath = '{}/samples_{}.npy'.format(sample_path, label)
		np.save(filepath, samples.cpu().data.numpy())

	@staticmethod
	def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Linear') != -1:
			m.weight.data.normal_(0.0, 0.02)
			m.bias.data.fill_(0)
		elif classname.find('Conv') != -1:
			torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)

	@staticmethod
	def normalize(mx):
		"""Row-normalize sparse matrix"""
		rowsum = np.array(mx.sum(1))
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.
		r_mat_inv = sp.diags(r_inv)
		mx = r_mat_inv.dot(mx)
		return mx

	################### Ops ##################
	@staticmethod
	def mkdir(name, rm=False):
		if not os.path.exists(name):
			os.makedirs(name)
		elif rm:
			shutil.rmtree(name)
			os.makedirs(name)

	@staticmethod
	def log(logf, msg, console_print=True):
		logf.write(msg + '\n')
		if console_print:
			print(msg)
