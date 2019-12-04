"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 10/07/2019
	Morton Autoregressive model
"""

import torch
from torch import nn

class VAE(nn.Module):
	def __init__(self, dim_input=1024, dim_mu=20):
		super(VAE, self).__init__()

		hidden = [dim_input//2, dim_input//4]
		self.model_encode = nn.Sequential(
			nn.Linear(dim_input, hidden[0]),
			nn.BatchNorm1d(hidden[0]),
			nn.ReLU(),
			nn.Linear(hidden[0], hidden[1]),
			nn.BatchNorm1d(hidden[1]),
			nn.ReLU(),
		)
		self.fc_mu = nn.Linear(hidden[1], dim_mu)
		self.fc_var = nn.Linear(hidden[1], dim_mu)

		self.model_decode = nn.Sequential(
			nn.Linear(dim_mu, hidden[1]),
			nn.ReLU(),
			nn.Linear(hidden[1], hidden[0]),
			nn.ReLU(),
			nn.Linear(hidden[0], dim_input),
			nn.Tanh()
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
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar
