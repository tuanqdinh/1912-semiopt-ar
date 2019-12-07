"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 10/07/2019
	Morton Autoregressive model
"""

import torch
from torch import nn
from vae.pointnet import PointNetCls, PointGen


class AE(nn.Module):
	def __init__(self, dim_input=1024, dim_embed=128):
		super(AE, self).__init__()

		self.model_encode = PointNetCls(input_dim=dim_input, output_dim=dim_embed)
		self.model_decode = PointGen(input_dim=dim_embed, output_dim=dim_input*3)
		self.relu = nn.ReLU()

	def encode(self, x):
		h1 = self.model_encode(x)
		return h1

	def decode(self, z):
		return self.model_decode(z)

	def forward(self, x):
		# x = x.view(-1, 1024 * 3) # shoul be in the training
		z = self.encode(x)
		return self.decode(z)

class AE2(nn.Module):
	def __init__(self, dim_input=1024, dim_embed=64, device='cuda'):
		super(AE2, self).__init__()
		self.dim_embed = dim_embed
		self.model_encode = PointNetCls(input_dim=dim_input, output_dim=dim_embed*2)
		# self.model_encode = PointNetCls(input_dim=dim_input, output_dim=dim_embed)
		# self.model_encode3 = PointNetCls(input_dim=dim_input, output_dim=dim_embed)
		# self.model_encode4 = PointNetCls(input_dim=dim_input, output_dim=dim_embed)

		dim_part = dim_input // 2
		self.model_decode1 = PointGen(input_dim=dim_embed, output_dim=dim_part*3)
		self.model_decode2 = PointGen(input_dim=dim_embed, output_dim=dim_part*3)
		# self.model_decode3 = PointGen(input_dim=dim_embed, output_dim=dim_part*3)
		# self.model_decode4 = PointGen(input_dim=dim_embed, output_dim=dim_part*3)

		# 3 transfromation
		self.pos = torch.randn([2, 3], dtype=torch.float, requires_grad=True).to(device)

	def encode(self, x):
		h = self.model_encode(x)
		h1 = h[:, :self.dim_embed]
		h2 = h[:, self.dim_embed:]
		# h3 = self.model_encode3(x)
		# h4 = self.model_encode4(x)
		return (h1, h2)

	def decode(self, z):
		h1, h2 = z
		y1 = self.model_decode1(h1).permute(0, 2, 1)
		y2 = self.model_decode2(h2).permute(0, 2, 1)
		# y3 = self.model_decode3(h3).permute(0, 2, 1)
		# y4 = self.model_decode4(h4).permute(0, 2, 1)
		y = (y1, y2)
		return y

	def translate(self, y):
		y1, y2 = y
		out1 = y1
		out2 = y2
		# out1 = y1 + self.pos[0, :]
		# out2 = y2 + self.pos[1, :]
		# out3 = y3 + self.pos[2, :]
		# out4 = y4 + self.pos[3, :]
		return torch.cat([out1, out2], dim=1)

	def forward(self, x):
		# x = x.view(-1, 1024 * 3) # shoul be in the training
		z = self.encode(x)
		y = self.decode(z)
		x_hat = self.translate(y)

		return x_hat, y
