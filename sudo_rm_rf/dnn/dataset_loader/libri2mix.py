"""!
@brief Pytorch dataloader for Libri2mix dataset.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import os
import numpy as np
import pickle
import glob2

import sys
current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(root_dir)

import sudo_rm_rf.dnn.dataset_loader.abstract_dataset as abstract_dataset
from scipy.io import wavfile
from tqdm import tqdm

EPS = 1e-8
enh_single = {'mixture': 'mix_single',
              'sources': ['s1', 'noise'],
              'n_sources': 1}
enh_single_white_noise = {
              'mixture': 'source_with_white_noise',
              'sources': ['s1', 'white_noise'],
              'n_sources': 1}
enh_both = {'mixture': 'mix_both',
            'sources': ['mix_clean', 'noise'],
            'n_sources': 1}
sep_clean = {'mixture': 'mix_clean',
             'sources': ['s1', 's2'],
             'n_sources': 2}
sep_noisy = {'mixture': 'mix_both',
             'sources': ['s1', 's2', 'noise'],
             'n_sources': 2}

WHAM_TASKS = {'enhance_single_white_noise': enh_single_white_noise,
              'enhance_single': enh_single,
              'enhance_both': enh_both,
              'sep_clean': sep_clean,
              'sep_noisy': sep_noisy}
WHAM_TASKS['enh_single'] = WHAM_TASKS['enhance_single']
WHAM_TASKS['enh_both'] = WHAM_TASKS['enhance_both']


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for Libri2mix source separation and speech enhancement
    tasks.

    Example of kwargs:
        root_dirpath='/mnt/data/wham', task='enh_single',
        split='tr', sample_rate=8000, timelength=4.0,
        normalize_audio=False, n_samples=0, zero_pad=False
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.task = self.get_arg_and_check_validness(
            'task', known_type=str, choices=WHAM_TASKS.keys())

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.normalize_audio = self.get_arg_and_check_validness(
            'normalize_audio', known_type=bool)

        self.min_or_max = self.get_arg_and_check_validness(
            'min_or_max', known_type=str, choices=['min', 'max'])

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['dev', 'test', 'train-100',
                                              'train-360'])

        self.n_samples = self.get_arg_and_check_validness(
            'n_samples', known_type=int, extra_lambda_checks=[lambda x: x >= 0])

        self.sample_rate = self.get_arg_and_check_validness('sample_rate',
                                                            known_type=int)
        self.root_path = self.get_arg_and_check_validness(
            'root_dirpath', known_type=str,
            extra_lambda_checks=[lambda y: os.path.lexists(y)])
        self.dataset_dirpath = self.get_path()

        self.mixtures_info_metadata_path = os.path.join(
            self.dataset_dirpath, 'metadata')

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.time_samples = int(self.sample_rate * self.timelength)

        # Create the indexing for the dataset
        mix_folder_path = os.path.join(self.dataset_dirpath,
                                       WHAM_TASKS[self.task]['mixture'])
        self.file_names = []
        self.available_mixtures = glob2.glob(mix_folder_path + '/*.wav')

        self.mixtures_info = []
        print('Parsing Dataset found at: {}...'.format(self.dataset_dirpath))
        if not os.path.lexists(self.mixtures_info_metadata_path):
            for file_path in tqdm(self.available_mixtures):
                sample_rate, waveform = wavfile.read(file_path)
                assert sample_rate == self.sample_rate
                numpy_wav = np.array(waveform)
                self.mixtures_info.append(
                    [os.path.basename(file_path), numpy_wav.shape[0]])

            print('Dumping metadata in: {}'.format(
                self.mixtures_info_metadata_path))
            with open(self.mixtures_info_metadata_path, 'wb') as filehandle:
                pickle.dump(self.mixtures_info, filehandle)

        if os.path.lexists(self.mixtures_info_metadata_path):
            with open(self.mixtures_info_metadata_path, 'rb') as filehandle:
                self.mixtures_info = pickle.load(filehandle)
                print('Loaded metadata from: {}'.format(
                    self.mixtures_info_metadata_path))

        self.file_names = [(path, n_samples)
                           for (path, n_samples) in self.mixtures_info
                           if (n_samples >= self.time_samples or self.zero_pad)]
        if self.n_samples > 0:
            self.file_names = self.file_names[:self.n_samples]

        max_time_samples = max([n_s for (_, n_s) in self.file_names])
        self.file_names = [x for (x, _) in self.file_names]

        # for the case that we need the whole audio input
        if self.time_samples <= 0.:
            self.time_samples = max_time_samples

    def get_path(self):
        path = os.path.join(self.root_path,
                            'wav{}k'.format(int(self.sample_rate / 1000)),
                            self.min_or_max, self.split)
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
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]

        mixture_path = os.path.join(self.dataset_dirpath,
                                    WHAM_TASKS[self.task]['mixture'],
                                    filename)
        _, waveform = wavfile.read(mixture_path)
        mixture_wav = np.array(waveform)
        mixture_wav = torch.tensor(mixture_wav, dtype=torch.float32)
        mixture_wav = self.safe_pad(mixture_wav)

        sources_list = []
        for source_name in WHAM_TASKS[self.task]['sources']:
            source_path = os.path.join(self.dataset_dirpath,
                                       source_name, filename)
            try:
                _, waveform = wavfile.read(source_path)
            except Exception as e:
                print(e)
                raise IOError('Could not load file from: {}'.format(
                    source_path))
            numpy_wav = np.array(waveform)
            source_wav = torch.tensor(numpy_wav, dtype=torch.float32)
            source_wav = self.safe_pad(source_wav)
            sources_list.append(source_wav)

        if self.normalize_audio:
            mix_std = mixture_wav.detach().cpu().numpy().std()
            mixture_wav = normalize_tensor_wav(mixture_wav, std=mix_std)
            sources_list = [normalize_tensor_wav(s, std=mix_std)
                            for s in sources_list]
        sources_wavs = torch.stack(sources_list, dim=0)

        return mixture_wav, sources_wavs

    def get_generator(self, batch_size=4, shuffle=True, num_workers=4):
        generator_params = {'batch_size': batch_size,
                            'shuffle': shuffle,
                            'num_workers': num_workers,
                            'drop_last': True}
        return torch.utils.data.DataLoader(self, **generator_params)


def test_generator():
    wham_root_p = '/mnt/data/libri_mix/Libri2Mix'
    batch_size = 3
    sample_rate = 8000
    timelength = 4.0
    time_samples = int(sample_rate * timelength)
    data_loader = Dataset(
        root_dirpath=wham_root_p, task='sep_clean',
        split='train-100', sample_rate=sample_rate, timelength=timelength,
        zero_pad=True, min_or_max='min',
        normalize_audio=False, n_samples=10)
    generator = data_loader.get_generator(batch_size=batch_size, num_workers=1)

    for mixture, sources in generator:
        print(mixture.shape)
        assert mixture.shape == (batch_size, time_samples)
        assert sources.shape == (batch_size, 2, time_samples)


    # test the testing set with batch size 1 only
    data_loader = Dataset(
        root_dirpath=wham_root_p, task='sep_clean',
        split='test', sample_rate=sample_rate, timelength=-1.,
        zero_pad=False, min_or_max='min',
        normalize_audio=False, n_samples=10)
    generator = data_loader.get_generator(batch_size=1, num_workers=1)

    for mixture, sources in generator:
        print(mixture.shape)

if __name__ == "__main__":
    test_generator()
