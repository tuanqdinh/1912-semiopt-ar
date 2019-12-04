"""
    @author Tuan Dinh tuandinh@cs.wisc.edu
    @date 08/14/2019
    Loading data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
"""
class Generator(nn.Module):
    def __init__(self, input_size=100, output_size=4225):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.PReLU())
            return layers

        self.model = nn.Sequential(
            *block(input_size, 32, normalize=False),
            *block(32, 64),
            nn.Linear(64, output_size),
        )

    def forward(self, z):
        return torch.tanh(self.model(z))

"""
"""
class Discriminator(nn.Module):
    def __init__(self, input_size=4225):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
