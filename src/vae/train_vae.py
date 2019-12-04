import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import argparse

from tools.Trainer import VAETrainer
from tools.PointCloudDataset import PointCloudDataset
from models.AutoEncoder import PointCloudVAE
from chamferdist import ChamferDistance
from EMD.emd import EmdDistance

parser = argparse.ArgumentParser(description='Point Cloud AE.')
parser.add_argument("-d", "--datapath", type=str, help="Dataset path.", default="")
parser.add_argument("-n", "--name", type=str, help="Name of the experiment", default="PointGen")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size", default=64)
parser.add_argument("-e", "--encSize", type=int, help="Encoding size", default=128)
parser.add_argument("-f", "--factorNoise", type=float, help="Noise factor", default=0.0)
parser.add_argument("-lr", "--lr", type=float, default=1e-5)
parser.add_argument("--train", dest='train', action='store_true')
parser.set_defaults(train=False)


# pc_datapath = "../../data/ShapeNet7/02691156_train.npy"
pc_datapath = "../../data/plane/hilbert_data_s3.npy"

if __name__ == '__main__':
    args = parser.parse_args()

    vae = PointCloudVAE(1024, 3, name=args.name, enc_size=args.encSize,
            batch_size=args.batchSize)
    #vae.load('checkpoint')
    optimizer = optim.Adam(vae.parameters(), lr=1e-5)

    # dataset = PointCloudDataset(args.datapath)
    data = np.load(pc_datapath)
    dataset = torch.Tensor(data)
    dataset = dataset.permute(0, 2, 1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
            shuffle=True, num_workers=2, drop_last = True)
    trainer = VAETrainer(vae, loader, optimizer, EmdDistance())
    net_path = 'checkpoint/{}/model-01000.pth'.format(args.name)
    vae.load_state_dict(torch.load(net_path))
    trainer.train(2000)
