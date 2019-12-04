"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""
import time
from utils.helper import Helper

import torch
import torch.autograd as autograd
import torch.nn.functional as F

def chamfer_batch(a, b):
	a = torch.transpose(a, 1, 2).contiguous()
	b = torch.transpose(b, 1, 2).contiguous()

	d = batch_pairwise_dist(a, b)
	dx = torch.min(d, dim=2)[0].sum(dim=1)
	dy = torch.min(d, dim=1)[0].sum(dim=1)
	return (dx + dy).sum()

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))

def bce(x, l):
	# [N, 3, 32, 32]
	x = x.permute(0, 2, 3, 1)
	# [N, 90, 32, 32]
	l = l.permute(0, 2, 3, 1)
	xs = [int(y) for y in x.size()]
	ls = [int(y) for y in l.size()]
	# here and below: unpacking the params of the mixture of logistics
	nr_mix = int(ls[-1] / 10)
	logit_probs = l[:, :, :, :nr_mix]
	l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
	means = l[:, :, :, :, :nr_mix]
	means = torch.sum(means, dim=-1)

	x_hat = means.view(xs[0], xs[1]*xs[2], 3)
	x = x.view(xs[0], xs[1]*xs[2], 3)

	ds = F.mse_loss(x_hat, x, reduction='sum')
	return ds

def emd_batch(x, l, loss_fn):
	# [N, 3, 32, 32]
	x = x.permute(0, 2, 3, 1)
	# [N, 90, 32, 32]
	l = l.permute(0, 2, 3, 1)
	xs = [int(y) for y in x.size()]
	ls = [int(y) for y in l.size()]
	# here and below: unpacking the params of the mixture of logistics
	nr_mix = int(ls[-1] / 10)
	logit_probs = l[:, :, :, :nr_mix]
	pi = log_prob_from_logits(logit_probs).exp()

	l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
	means = l[:, :, :, :, :nr_mix]
	t = pi.unsqueeze(3).expand(means.shape)
	means = torch.sum(means * t, dim=-1)

	means = means.view(xs[0], xs[1]*xs[2], 3)
	x = x.view(xs[0], xs[1]*xs[2], 3)

	ds = loss_fn(x, means)
	return ds

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x_hat, x, mu, logvar):
	# BCE = F.binary_cross_entropy(x_hat, x.view(-1, 1024 *3), reduction='sum')
	# MSE = F.mse_loss(x_hat, x, reduction='sum')
	# see Appendix B from VAE paper:
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	# return MSE + KLD
	recon = torch.norm(x_hat - x, p=2).mean()

	return recon, KLD

def recon_loss(x_hat, x):
		# return MSE + KLD
	BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
	# recon = torch.norm(x_hat - x, p=2).mean()
	return BCE

def kld(mu, logvar):
	# BCE = F.binary_cross_entropy(x_hat, x.view(-1, 1024 *3), reduction='sum')
	# MSE = F.mse_loss(x_hat, x, reduction='sum')
	# see Appendix B from VAE paper:
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return KLD


def batch_pairwise_dist(x, y):
	bs, num_points, points_dim = x.size()
	xx = torch.bmm(x, x.transpose(2,1))
	yy = torch.bmm(y, y.transpose(2,1))
	zz = torch.bmm(x, y.transpose(2,1))
	diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
	rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
	ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
	P = (rx.transpose(2,1) + ry - 2*zz)
	return P
