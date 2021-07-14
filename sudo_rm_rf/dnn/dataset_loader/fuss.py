"""!
@brief Pytorch dataloader for fuss dataset.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import os
import numpy as np
import pickle
import glob2
import sudo_rm_rf.dnn.dataset_loader.abstract_dataset as \
    abstract_dataset
from scipy.io import wavfile
from tqdm import tqdm
from time import time

EPS = 1e-8


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for FUSS source separation and speech enhancement tasks.

    Example of kwargs:
        root_dirpath='/mnt/data/fuss',
        split='train', sample_rate=8000, timelength=4.0,
        normalize_audio=False, n_samples=0, zero_pad=False
    """

    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        # add num_sources limit
        self.max_num_sources = self.get_arg_and_check_validness(
            'max_num_sources', known_type=int,
            extra_lambda_checks=[lambda x: (x >= 1) and (x <= 4)])

        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)

        self.min_num_sources = self.get_arg_and_check_validness(
            'min_num_sources', known_type=int,
            extra_lambda_checks=[
                lambda x: (x >= 1) and (x <= self.max_num_sources)])

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.normalize_audio = self.get_arg_and_check_validness(
            'normalize_audio', known_type=bool)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['train', 'eval', 'validation'])

        self.sample_rate = self.get_arg_and_check_validness(
            'sample_rate', known_type=int, choices=[16000])

        self.root_path = self.get_arg_and_check_validness(
            'root_dirpath', known_type=str,
            extra_lambda_checks=[lambda y: os.path.lexists(y)])
        self.dataset_dirpath = self.get_path()

        self.mixtures_info_metadata_path = os.path.join(
            self.dataset_dirpath, 'metadata')

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.time_samples = int(self.sample_rate * self.timelength)

        # Create the indexing for the dataset by reading the corresponding txt
        metadata_path = os.path.join(self.root_path, self.split +
                                     '_example_list.txt')
        with open(metadata_path, 'r') as f:
            lines = f.readlines()
        self.available_sources = [
            (l.split()[1:], len(l.split()) - 1) for l in lines]

        # Get the examples with the appropriate number of sources
        self.source_folder_names = []
        for sources_paths, n_sources in tqdm(self.available_sources):
            if self.min_num_sources <= n_sources <= self.max_num_sources:
                self.source_folder_names.append(
                    [os.path.join(self.root_path, sp)
                     for sp in sorted(sources_paths)])

        self.actual_n_samples = len(self.source_folder_names)
        self.n_samples = self.get_arg_and_check_validness(
            'n_samples', known_type=int, extra_lambda_checks=[
                lambda x: (x >= 0) and (x <= self.actual_n_samples)])
        if self.n_samples > 0:
            self.source_folder_names = self.source_folder_names[:self.n_samples]
        else:
            self.n_samples = len(self.source_folder_names)

    def get_path(self):
        path = os.path.join(self.root_path, self.split)
        if os.path.lexists(path):
            return path
        else:
            raise IOError('Dataset path: {} not found!'.format(path))

    def safe_pad(self, tensor_wav):
        if self.zero_pad and tensor_wav.shape[0] < self.time_samples:
            appropriate_shape = tensor_wav.shape
            padded_wav = torch.zeros(
                list(appropriate_shape[:-1]) + [self.time_samples],
                dtype=torch.float32)
            padded_wav[:tensor_wav.shape[0]] = tensor_wav
            return padded_wav[:self.time_samples]
        else:
            return tensor_wav[:self.time_samples]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sources_folder_name = self.source_folder_names[idx]
        actual_n_sources = len(sources_folder_name)

        sources_list = []
        for source_wav_path in sources_folder_name:
            _, waveform = wavfile.read(source_wav_path)
            max_len = len(waveform)
            rand_start = 0
            if self.time_samples > 0:
                if self.augment and max_len > self.time_samples:
                    the_time = int(np.modf(time())[0] * 100000000)
                    np.random.seed(the_time)
                    rand_start = np.random.randint(
                        0, max_len - self.time_samples)
                waveform = waveform[rand_start:rand_start + self.time_samples]

            source_wav = np.array(waveform)
            source_wav = torch.tensor(source_wav, dtype=torch.float32)
            sources_list.append(source_wav)

        sources_wavs = torch.stack(sources_list, dim=0)
        zero_padded_sources_wavs = torch.zeros(
            (self.max_num_sources, sources_wavs.shape[-1]), dtype=torch.float32)
        zero_padded_sources_wavs[:actual_n_sources] = sources_wavs

        return zero_padded_sources_wavs

    def get_generator(self, batch_size=4, shuffle=True, num_workers=4):
        generator_params = {'batch_size': batch_size,
                            'shuffle': shuffle,
                            'num_workers': num_workers,
                            'drop_last': True}
        return torch.utils.data.DataLoader(self, **generator_params,
                                           pin_memory=True)


def test_generator():
    fuss_root_p = '/mnt/data/FUSS/fuss_dev/ssdata'
    batch_size = 3
    sample_rate = 16000
    timelength = 5.0
    time_samples = int(sample_rate * timelength)
    max_num_sources = 3
    min_num_sources = 1
    data_loader = Dataset(
        root_dirpath=fuss_root_p, max_num_sources=max_num_sources,
        min_num_sources = min_num_sources,
        split='train', sample_rate=sample_rate, timelength=timelength,
        zero_pad=True, augment=False, normalize_audio=False, n_samples=12)
    generator = data_loader.get_generator(batch_size=batch_size, num_workers=1)

    for sources in generator:
        assert sources.shape == (batch_size, max_num_sources, time_samples)

    # test the testing set with batch size 1 only
    data_loader = Dataset(
        root_dirpath=fuss_root_p, max_num_sources=4,
        min_num_sources = 4,
        split='eval', sample_rate=sample_rate, timelength=-1.,
        zero_pad=False,
        normalize_audio=False, n_samples=10, augment=False)

    generator = data_loader.get_generator(batch_size=1, num_workers=1)

    for sources in generator:
        assert sources.shape == (1, 4, 160000)


if __name__ == "__main__":
    test_generator()
