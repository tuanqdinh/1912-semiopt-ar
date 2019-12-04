import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Model import Model
from tools import Ops
from .AutoEncoder import NormalReg, MultiResConv1d, MultiResConvTranspose1d
from tools.PointCloudDataset import save_objs
from tools.plotter import Plotter

class MultiResVAE(Model):

	def __init__(self, size, dim, batch_size=64, enc_size=100, kernel_size=2,
			reg_fn=NormalReg(),
			noise = 0,
			name="MLVAE"):
		super(MultiResVAE, self).__init__(name)

		self.reg_fn = reg_fn

		self.size = size
		self.dim = dim
		self.enc_size = enc_size
		self.batch_size = batch_size
		self.kernel_size = kernel_size
		self.enc_modules = nn.ModuleList()
		self.dec_modules = nn.ModuleList()
		self.upsample = Ops.NNUpsample1d()
		self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
		self.noise_factor = noise

		self.enc_noise = torch.FloatTensor(self.batch_size, self.enc_size)

		custom_nfilters = [3, 32, 64, 128, 256, 512, 512]
		custom_nfilters = np.array(custom_nfilters)
		custom_nfilters[1:] = custom_nfilters[1:]/2
		self.last_size = 16

		self.noise = torch.FloatTensor(self.batch_size, self.enc_size)

		current_size = self.size
		layer_num = 1
		padding = (self.kernel_size - 1)/2
		n_channels = []
		n_channels.append(custom_nfilters[layer_num-1])
		while current_size > self.last_size:
			in_channels = custom_nfilters[layer_num-1]
			out_channels = custom_nfilters[layer_num]
			conv_enc = MultiResConv1d('down{}'.format(layer_num),
					in_channels, out_channels)
			current_size /= 2
			in_channels = out_channels
			n_channels.append(out_channels)
			layer_num += 1

			self.enc_modules.append(conv_enc)

		self.enc_fc = nn.Linear(3*self.last_size*in_channels, self.enc_size)
		#self.enc_fc_mean = nn.Linear(3*self.last_size*in_channels, self.enc_size)
		#self.enc_fc_var = nn.Linear(3*self.last_size*in_channels, self.enc_size)
		self.dec_fc = nn.Linear(self.enc_size, self.last_size*n_channels[-1])

		self.final_feature = 128
		n_channels.reverse()
		n_channels[-1] = self.final_feature
		current_size = self.last_size
		layer_num = 1
		padding = (self.kernel_size - 1)/2
		while current_size < self.size:
			in_channels = n_channels[layer_num-1]
			out_channels = n_channels[layer_num]
			conv_dec = MultiResConvTranspose1d('up{}'.format(layer_num),
					in_channels, out_channels)
			current_size *= 2
			in_channels = out_channels
			layer_num += 1

			self.dec_modules.append(conv_dec)

		self.final_conv = nn.Sequential()
		self.final_conv.add_module('final_conv1',
				nn.ConvTranspose1d(self.final_feature*3, 128, 1, stride=1, padding=0))
		self.final_conv.add_module('bn_final',
				nn.BatchNorm1d(128))
		self.final_conv.add_module('relu_final',
				nn.ReLU(inplace=True))
		self.final_conv.add_module('final_conv2',
				nn.ConvTranspose1d(128, 3, 1, stride=1, padding=0))
		self.final_conv.add_module('tanh_final',
				nn.Tanh())


	def enc_forward(self, x):
		x0 = x
		x1 = self.pool(x)
		x2 = self.pool(x1)

		enc_tensors = []
		enc_tensors.append([x0, x1, x2])

		for enc_op in self.enc_modules:
			enc_tensors.append(enc_op(enc_tensors[-1]))

		t0 = enc_tensors[-1][0]
		t1 = self.upsample(enc_tensors[-1][1])
		t2 = self.upsample(self.upsample(enc_tensors[-1][2]))
		t = torch.cat((t0, t1, t2), 1).view(self.batch_size, -1)

		encoding = self.enc_fc(t)
		return encoding, enc_tensors
		#encoding_mean = self.enc_fc_mean(t)
		#encoding_var = self.enc_fc_var(t)
		#return (encoding_mean, encoding_var)


	def dec_forward(self, x):

		mr_enc0 = self.dec_fc(x).view(self.batch_size, -1, self.last_size)
		mr_enc1 = self.pool(mr_enc0)
		mr_enc2 = self.pool(mr_enc1)
		mr_enc = [mr_enc0, mr_enc1, mr_enc2]

		dec_tensors = []
		dec_tensors.append(mr_enc)

		for i in range(0, len(self.dec_modules)-1):
			dec_tensors.append(self.dec_modules[i](dec_tensors[-1]))

		conv_out = self.dec_modules[-1](dec_tensors[-1])
		out0 = conv_out[0]
		out1 = self.upsample(conv_out[1])
		out2 = self.upsample(self.upsample(conv_out[2]))
		out = torch.cat((out0, out1, out2), 1)
		return self.final_conv(out)

#
#    def reparameterize(self, mu, logvar):
#        if self.training:
#          std = logvar.mul(0.5).exp_()
#          eps = Variable(std.data.new(std.size()).normal_())
#          return eps.mul(std).add_(mu)
#        else:
#          return mu
#

	def forward(self, x):
		encoding = self.enc_forward(x)[0]
		self.enc_noise.normal_()

		added_noise = Variable(self.noise_factor*self.enc_noise.cuda())

		encoding += added_noise
		return self.dec_forward(encoding)


	def encoding_regularizer(self, x):
		return self.reg_fn(self.enc_forward(x)[0])


	def sample(self):
		self.noise.normal_()
		return self.dec_forward(Variable(self.noise.cuda()))


	def save_results(self, path, data, start_idx=0, name='sample'):
		results = data.cpu().data.numpy()
		results = results.transpose(0, 2, 1)

		# xs = results.cpu().data.numpy()
		for i in range(10):
			img = results[i, :, :]
			filepath = path + '/{}-{}.png'.format(name, i)
			Plotter.plot_pc(img, filepath)

		# save_objs(results, path, start_idx)
		print("Points saved.")
