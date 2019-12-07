"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""
import os
import torch
import numpy as np
from utils.sfc import SFC
# from sfc import SFC


class Provider:

	@staticmethod
	def load_data(dataset_path, mode_space, m, p, normalized=False, renew=True):
		fpath_hilbert = os.path.join(dataset_path, 'hilbert_data_s{}.npy'.format(mode_space))
		# load old data
		if os.path.isfile(fpath_hilbert) and not(renew):
			print('Load hilbert codes')
			sorted_data = np.load(fpath_hilbert)
		else:
			print("Load raw data")
			fpath_pc = os.path.join(dataset_path, 'data.npy')
			points = np.load(fpath_pc)
			# convert data to hilbertcode
			seqs = SFC.encode_sfc(points, M=m, p=p)
			# store
			if mode_space > 1:
				ind = np.argsort(seqs, axis=1)
				sorted_data = np.zeros(points.shape)
				for i in range(ind.shape[0]):
					row = points[i, ind[i, :], :]
					sorted_data[i, :, :] = row
				fpath_m= os.path.join(dataset_path, 'hilbert_data_s{}.npy'.format(m))
				np.save(fpath_m, sorted_data)
			else:
				sorted_data = np.sort(seqs, axis=1)
				fpath_hilbert = os.path.join(dataset_path, 'hilbert_data_s1.npy')
				np.save(fpath_hilbert, sorted_data)

		return sorted_data


	@staticmethod
	def load_mnist(dataset_path, normalized=False, renew=True):
		print('MNIST')
		fname_hilbert = 'hilbert_data.npy'
		fpath_hilbert = os.path.join(dataset_path, fname_hilbert)
		if os.path.isfile(fpath_hilbert) and not(renew):
			print('Load hilbert codes')
			seqs = np.load(fpath_hilbert)
		else:
			print("Load data")
			fname_pc = 'data.npy'
			fpath_pc = os.path.join(dataset_path, fname_pc)
			points = np.load(fpath_pc, allow_pickle=True)
			points = points.item()
			points = points['X_train']
			# convert data to hilbertcode
			seqs = SFC.encode_sfc(points)
			np.save(fpath_hilbert, seqs)
		# Tensor
		if normalized:
			seqs = seqs * 2 - 1 # [-1, 1]
		tseqs = torch.Tensor(seqs)
		return tseqs

	@staticmethod
	def generate_square(nsamples=5000, d=128):
		boundary = [0.1, 0.9]
		eps = 0.001
		points = []
		s = d //4
		for i in range(2):
			# x
			x = boundary[i]
			px = np.random.uniform(low=x-eps, high=x+eps, size=(nsamples, s))
			py = np.random.uniform(low=0, high=1, size=(nsamples, s))
			ps = np.stack([px, py], axis=2)
			if i == 0:
				points = ps
			else:
				points = np.concatenate([points, ps], axis=1)

		for i in range(2):
			# x
			y = boundary[i]
			px = np.random.uniform(low=0, high=1, size=(nsamples, s))
			py = np.random.uniform(low=y-eps, high=y+eps, size=(nsamples, s))
			ps = np.stack([px, py], axis=2)
			points = np.concatenate([points, ps], axis=1)


		np.save('../../data/toy/data.npy', points)
		return points

	@staticmethod
	def generate_line(nsamples=5000, d=128):
		boundary = [0.1, 0.9]
		eps = 0.001
		px = np.random.uniform(low=0.1, high=0.9, size=(nsamples, d))
		py = px + np.random.uniform(low=-eps, high=eps, size=(nsamples, d))
		points = np.stack([px, py], axis=2)

		np.save('../../data/toy/line.npy', points)
		return points
