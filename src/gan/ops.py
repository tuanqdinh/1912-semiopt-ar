"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""
import time
from utils.helper import Helper

import torch
import torch.autograd as autograd


def log_loss_gan(logf, epoch, num_epochs, step, total_step, D_cost, G_cost,  start_time):
	# convert
	loss_d = D_cost.cpu().data.numpy()
	loss_g = G_cost.cpu().data.numpy()
	# msg
	message = 'Epoch [{}/{}], Step [{}/{}], LossD: {:.4f}, LossG: {:.4f}, time: {:.4f}s'.format(epoch, num_epochs, step, total_step, loss_d, loss_g, time.time() - start_time)
	# log out
	Helper.log(logf, message)

def get_gradient_penalty(netD, real_data, fake_data):
	#
	alpha = torch.rand(real_data.shape[0], 1, 1).to(Helper.__device__)
	alpha = alpha.expand(real_data.shape)
	#
	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
	interpolates = interpolates.to(Helper.__device__)
	interpolates = autograd.Variable(interpolates, requires_grad=True)
	#
	disc_interpolates = netD(interpolates)
	#
	torch_ones = torch.ones(disc_interpolates.size()).to(Helper.__device__)
	grads = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch_ones, create_graph=True, retain_graph=True, only_inputs=True)
	grad = grads[0]
	# penalty
	grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
	return grad_penalty

def generate_sample(net, nsamples, embed_size):
    noise = torch.randn(nsamples, embed_size).to(Helper.__device__)
    with torch.no_grad():
        noisev = autograd.Variable(noise)
    return net(noisev)
