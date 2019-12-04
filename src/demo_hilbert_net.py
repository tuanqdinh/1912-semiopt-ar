"""
    Demo if a neural net can learn sfc
"""

import torch
import torch.nn as nn
import torch.optim as optim

import os, sys
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--result_path', type=str, default='../result', help='output path')
parser.add_argument('--flag_retrain', default=False, action='store_true', help='Re train')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

net_path = os.path.join(args.result_path, 'sfc.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HillbertNet(nn.Module):
    '''
        receive N, p
        output the score/sorted sequence
    '''
    def __init__(self, N, p):
        super(HillbertNet, self).__init__()

        self.ndim = N
        self.num_points = 2 ** (self.ndim * p) - 1
        self.grid_size = 2 ** p - 1
        self.hilbert_curve = HilbertCurve(p, N)
        self.bin_mask = torch.Tensor(np.asarray([2**(N*p - 1 - k) for k in range(N*p)])).to(device)

        self.model = nn.Sequential(
            nn.Linear(self.ndim, self.grid_size),
            nn.BatchNorm1d(self.grid_size),
            nn.ReLU(),
            nn.Linear(self.grid_size, self.num_points),
            nn.BatchNorm1d(self.num_points),
            nn.ReLU(),
            nn.Linear(self.num_points, 1),
        )

    def generate_data(self, nsamples):
        data = np.zeros((nsamples, self.ndim + 1))
        x = np.random.randint(low=0, high=self.grid_size+1, size=(nsamples, self.ndim))
        for i in range(nsamples):
            coords = x[i, :]
            dist = self.hilbert_curve.distance_from_coordinates(coords)
            data[i, :-1] = coords
            data[i, -1] = dist/self.num_points

        return torch.Tensor(data)

    def forward(self, x):
        out = self.model(x)
        # out = torch.mv(out, self.bin_mask)
        # out = out /self.num_points
        return out

net = HillbertNet(N=2, p=1).to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
# criterion = nn.MSELoss()

if os.path.isfile(net_path) and args.flag_retrain:
	print('Load existing models')
	net.load_state_dict(torch.load(net_path))

# self-training
print("Training")
eps = 1e-5
lossfs = []
for e in range(args.num_epochs):
    # randomly generate some inputs
    batch_data = net.generate_data(nsamples=args.batch_size*100).to(device)
    input = batch_data[:, :-1]
    target = batch_data[:, -1]

    # encode into sfc score
    output = net(input)
    loss = (output - target)/(target + eps)
    loss = torch.abs(loss).mean()
    # loss = torch.mean((torch.log(output + eps)-torch.log(target + eps))**2)

    lossf = loss.data.item()
    lossfs.append(lossf)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("epoch {} average loss: {:.4f}".format(e, np.mean(lossfs)))

print('evaluation')
# evaluation
test_data = net.generate_data(nsamples=args.batch_size).to(device)
input = test_data[:, :-1]
target = test_data[:, -1:]
output = net(input)
result = torch.cat([test_data, output], dim=1)
print(result.data.cpu())
torch.save(net.state_dict(), net_path)
