
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Model import Model
import tools.DataVis as DataVis
from tools.PointCloudDataset import save_objs
from tools import Ops
from modules.nnd import NNDModule

class ChamferLoss(nn.Module):

	def __init__(self, n_samples=1024, cuda_opt=True):
		super(ChamferLoss, self).__init__()
		self.n_samples = n_samples
		self.dist = NNDModule()
		self.cuda_opt = cuda_opt


	def chamfer(self, a, b):
		pcsize = a.size()[1]

		a = torch.t(a)
		b = torch.t(b)
		mma = torch.stack([a]*pcsize)
		mmb = torch.stack([b]*pcsize).transpose(0,1)
		d = torch.sum((mma-mmb)**2,2).squeeze()

		return torch.min(d, 1)[0].sum() + torch.min(d, 0)[0].sum()


	def chamfer_batch(self, a, b):
		pcsize = a.size()[-1]

		if pcsize != self.n_samples:
			indices = np.arange(pcsize).astype(int)
			np.random.shuffle(indices)
			indices = torch.from_numpy(indices[:self.n_samples]).cuda()
			a = a[:, :, indices]
			b = b[:, :, indices]

		a = torch.transpose(a, 1, 2).contiguous()
		b = torch.transpose(b, 1, 2).contiguous()

		if self.cuda_opt:
			d1, d2 = self.dist(a, b)
			out = torch.sum(d1) + torch.sum(d2)
			return out
		else:
			d = Ops.batch_pairwise_dist(a, b)
			return torch.min(d, dim=2)[0].sum() + torch.min(d, dim=1)[0].sum()

		#mma = torch.stack([a]*self.n_samples, dim=1)
		#mmb = torch.stack([b]*self.n_samples, dim=1).transpose(1,2)
		#d = torch.sum((mma-mmb)**2,3).squeeze()
		#d = pd



	def forward(self, a, b):
		batch_size = a.size()[0]
		assert(batch_size == b.size()[0])
		loss = self.chamfer_batch(a, b)
		return loss/(float(batch_size) * 1)


class MultiResChamferLoss(nn.Module):

	def __init__(self, n_samples=1024):
		super(MultiResChamferLoss, self).__init__()
		self.n_samples = n_samples
		self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

	def chamfer_batch(self, a, b):
		pcsize = a.size()[-1]

		if pcsize > self.n_samples:
			pcsize = self.n_samples

			indices = np.arange(pcsize)
			np.random.shuffle(indices)
			indices = indices[:self.n_samples]

			a = a[:, :, indices]
			b = b[:, :, indices]

		a = torch.transpose(a, 1, 2)
		b = torch.transpose(b, 1, 2)
		#d = Ops.batch_pairwise_dist(a, b)
		mma = torch.stack([a]*self.n_samples, dim=1)
		mmb = torch.stack([b]*self.n_samples, dim=1).transpose(1,2)
		d = torch.sum((mma-mmb)**2,3).squeeze()
		#d = pd

		return (torch.min(d, dim=2)[0].sum() + torch.min(d, dim=1)[0].sum())/float(pcsize)


	def forward(self, a, b):
		batch_size = a[0].size()[0]

		b_samples = []
		b_samples.append(b)
		b_samples.append(self.pool(b_samples[-1]))
		b_samples.append(self.pool(b_samples[-1]))

		loss = 0.0
		for i in [0]:
			loss += self.chamfer_batch(a[i], b_samples[i])

		return 1e3*loss/float(batch_size)



class ChamferWithNormalLoss(nn.Module):

	def __init__(self, normal_weight=0.001, n_samples=1024):
		super(ChamferWithNormalLoss, self).__init__()
		self.normal_weight = normal_weight
		self.nlogger = DataVis.LossLogger("normal component")
		self.n_samples = n_samples


	def forward(self, a, b):
		pcsize = a.size()[-1]

		if pcsize != self.n_samples:
			indices = np.arange(pcsize)
			np.random.shuffle(indices)
			indices = indices[:self.n_samples]
			a = a[:, :, indices]
			b = b[:, :, indices]

		a_points = torch.transpose(a, 1, 2)[:, :, 0:3]
		b_points = torch.transpose(b, 1, 2)[:, :, 0:3]
		pd = Ops.batch_pairwise_dist(a_points, b_points)
		#mma = torch.stack([a_points]*self.n_samples, dim=1)
		#mmb = torch.stack([b_points]*self.n_samples, dim=1).transpose(1,2)
		d = pd

		a_normals = torch.transpose(a, 1, 2)[:, :, 3:6]
		b_normals = torch.transpose(b, 1, 2)[:, :, 3:6]
		mma = torch.stack([a_normals]*self.n_samples, dim=1)
		mmb = torch.stack([b_normals]*self.n_samples, dim=1).transpose(1,2)
		d_norm = 1 - torch.sum(mma*mmb,3).squeeze()
		d += self.normal_weight * d_norm

		normal_min_mean = torch.min(d_norm, dim=2)[0].mean()
		self.nlogger.update(normal_min_mean)

		chamfer_sym = torch.min(d, dim=2)[0].sum() + torch.min(d, dim=1)[0].sum()
		chamfer_sym /= a.size()[0]

		return chamfer_sym


class SampleChamfer(nn.Module):

	def __init__(self, normal_weight=0.001, n_samples=1024):
		super(SampleChamfer, self).__init__()
		self.normal_weight = normal_weight
		self.nlogger = DataVis.LossLogger("normal component")
		self.n_samples = n_samples


	def chamfer(self, a, b):

		a_indices = np.arange(a.size()[-1])
		b_indices = np.arange(b.size()[-1])
		np.random.shuffle(a_indices)
		np.random.shuffle(b_indices)
		a_indices = a_indices[:self.n_samples]
		b_indices = b_indices[:self.n_samples]
		a = a[:, a_indices]
		b = b[:, b_indices]

		a = torch.t(a)
		b = torch.t(b)
		mma = torch.stack([a[:, 0:3]]*self.n_samples)
		mmb = torch.stack([b[:, 0:3]]*self.n_samples).transpose(0,1)
		d = torch.sum((mma-mmb)**2,2).squeeze()

		#return torch.min(d, 1)[0].sum() + torch.min(d, 0)[0].sum()
		return torch.min(d, 0)[0].sum()

	def forward(self, a, b):
#        pcsize = a.size()[-1]
#
#        if pcsize != self.n_samples:
#            indices = np.arange(pcsize)
#            np.random.shuffle(indices)
#            indices = indices[:self.n_samples]
#            a = a[:, :, indices]
#            b = b[:, :, indices]
#
#        a_points = torch.transpose(a, 1, 2)[:, :, 0:3]
#        b_points = torch.transpose(b, 1, 2)[:, :, 0:3]
#        mma = torch.stack([a_points]*self.n_samples, dim=1)
#        mmb = torch.stack([b_points]*self.n_samples, dim=1).transpose(1,2)
#        d = torch.sum((mma-mmb)**2,3).squeeze()
#
#        a_normals = torch.transpose(a, 1, 2)[:, :, 3:6]
#        b_normals = torch.transpose(b, 1, 2)[:, :, 3:6]
#        mma = torch.stack([a_normals]*self.n_samples, dim=1)
#        mmb = torch.stack([b_normals]*self.n_samples, dim=1).transpose(1,2)
#        d_norm = 1 - torch.sum(mma*mmb,3).squeeze()
#        d += self.normal_weight * d_norm
#
#        normal_min_mean = torch.min(d_norm, dim=2)[0].mean()
#        self.nlogger.update(normal_min_mean)
#
#        chamfer_sym = torch.min(d, dim=2)[0].sum() + torch.min(d, dim=1)[0].sum()
#        chamfer_sym /= a.size()[0]

		return self.chamfer(a, b)



class SinkhornLoss(nn.Module):

	def __init__(self, n_iter=20, eps=1.0, batch_size=64, enc_size=512):
		super(SinkhornLoss, self).__init__()
		self.eps = eps
		self.n_iter = n_iter
		self.batch_size = batch_size
		self.normal_noise = torch.FloatTensor(batch_size, enc_size)


	def forward(self, x):
		bsize = x.size()[0]
		assert bsize == self.batch_size

		self.normal_noise.normal_()
		y = Variable(self.normal_noise.cuda())

		#Computes MSE cost
		mmx = torch.stack([x]*bsize)
		mmy = torch.stack([x]*bsize).transpose(0, 1)
		c = torch.sum((mmx-mmy)**2,2).squeeze()

		k = (-c/self.eps).exp()
		b = Variable(torch.ones((bsize, 1))).cuda()
		a = Variable(torch.ones((bsize, 1))).cuda()

		#Sinkhorn iterations
		for l in range(self.n_iter):
			a = Variable(torch.ones((bsize, 1))).cuda() / (torch.mm(k, b))
			b = Variable(torch.ones((bsize, 1))).cuda() / (torch.mm(k.t(), a))

		loss = torch.mm(k * c, b)
		loss = torch.sum(loss*a)
		return loss



class GeodesicChamferLoss(nn.Module):

	def __init__(self):
		super(GeodesicChamferLoss, self).__init__()


	def forward(self, a, b):
		pass


class L2WithNormalLoss(nn.Module):

	def __init__(self):
		super(L2WithNormalLoss, self).__init__()
		self.nlogger = DataVis.LossLogger("normal w/ L2")
		self.L1 = nn.L1Loss()

	def forward(self, a, b):
		position_loss = self.L1(a[:, 0:3, :], b[:, 0:3, :])
		normal_loss = torch.mean(1 - Ops.cosine_similarity(a[:, 3:6, :], b[:, 3:6, :]))
		self.nlogger.update(normal_loss)

		return normal_loss
