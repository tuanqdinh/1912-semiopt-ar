"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 10/07/2019
	Morton Autoregressive model
"""

import torch
from torch import nn
from models.pointnet import PointNetCls, PointGen

class VAE(nn.Module):
	def __init__(self, dim_input=1024, dim_embed=128, dim_mu=20):
		super(VAE, self).__init__()

		self.model_encode = PointNetCls(num_points=dim_input, k=dim_embed)
		self.model_decode = PointGen(num_points=dim_input, k=dim_mu)
		self.fc_mu = nn.Linear(dim_embed, dim_mu)
		self.fc_var = nn.Linear(dim_embed, dim_mu)
		self.sig = nn.Sigmoid()
		self.relu = nn.ReLU()

	def encode(self, x):
		h1 = self.model_encode(x)
		return self.sig(self.fc_mu(h1)), self.relu(self.fc_var(h1))

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, z, latent=False):
		return self.model_decode(z, latent)

	def forward(self, x):
		# x = x.view(-1, 1024 * 3) # shoul be in the training
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar


class VAE_classic(nn.Module):
	def __init__(self, dim_input=1024, dim_embed=128, dim_mu=20):
		super(VAE_classic, self).__init__()

		self.model_encode = nn.Sequential(
			nn.Linear(dim_input*3, 1000),
			nn.ReLU(),
			# nn.BatchNorm1d(100)
			nn.Linear(1000, 400),
			nn.ReLU(),
		)
		self.fc_mu = nn.Linear(400, dim_mu)
		self.fc_var = nn.Linear(400, dim_mu)

		self.model_decode = nn.Sequential(
			nn.Linear(dim_mu, 400),
			nn.ReLU(),
			nn.Linear(400, 1000),
			nn.ReLU(),
			nn.Linear(1000, dim_input*3),
			nn.Sigmoid()
		)

	def encode(self, x):
		h1 = self.model_encode(x)
		return self.fc_mu(h1), self.fc_var(h1)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, z):
		return self.model_decode(z)

	def forward(self, x):
		mu, logvar = self.encode(x.view(-1, 1024 * 3))
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar
