"""
Show raw audio and mu-law encode samples to make input source
"""
import os

import librosa
import numpy as np

import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, data_dir, sample_rate=16000, in_channels=256, trim=True):
        super(Dataset, self).__init__()

        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.trim = trim

        self.root_path = data_dir
        self.filenames = [x for x in sorted(os.listdir(data_dir))]

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames[index])


        return encoded_audio

    def __len__(self):
        return len(self.filenames)


class DataLoader(data.DataLoader):
    def __init__(self, data_dir, receptive_fields,
                 sample_size=0, sample_rate=16000, in_channels=256,
                 batch_size=1, shuffle=True):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
        :param sample_size: integer. number of timesteps to train at once.
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param sample_rate: sound sampling rates
        :param in_channels: number of input channels
        :param batch_size:
        :param shuffle:
        """
        dataset = Dataset(data_dir, sample_rate, in_channels)

        super(DataLoader, self).__init__(dataset, batch_size, shuffle)

        if sample_size <= receptive_fields:
            raise Exception("sample_size has to be bigger than receptive_fields")

        self.sample_size = sample_size
        self.receptive_fields = receptive_fields

        self.collate_fn = self._collate_fn

    def calc_sample_size(self, audio):
        return self.sample_size if len(audio[0]) >= self.sample_size\
                                else len(audio[0])

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    def _collate_fn(self, audio):
        audio = np.pad(audio, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')

        if self.sample_size:
            sample_size = self.calc_sample_size(audio)

            while sample_size > self.receptive_fields:
                inputs = audio[:, :sample_size, :]
                targets = audio[:, self.receptive_fields:sample_size, :]

                yield self._variable(inputs),\
                      self._variable(one_hot_decode(targets, 2))

                audio = audio[:, sample_size-self.receptive_fields:, :]
                sample_size = self.calc_sample_size(audio)
        else:
            targets = audio[:, self.receptive_fields:, :]
            return self._variable(audio),\
                   self._variable(one_hot_decode(targets, 2))
