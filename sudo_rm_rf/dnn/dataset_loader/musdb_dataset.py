"""!
@brief Pytorch dataloader for Musdb dataset.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import musdb as musdb_lib
import os
import torch
import numpy as np
from time import time
from tqdm import tqdm
from __config__ import MUSDBWAV_ROOT_PATH
from __config__ import MUSDBWAV8K_ROOT_PATH
from __config__ import MUSDB_ROOT_PATH
import sudo_rm_rf.dnn.dataset_loader.abstract_dataset as \
    abstract_dataset


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for MUSDB source separation single or stereo.

    Example of kwargs:
        root_dirpath='/mnt/data/wham', task='enh_single',
        split='tr', sample_rate=8000, timelength=4.0,
        normalize_audio=False, n_samples=0, zero_pad=False
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.n_channels = self.get_arg_and_check_validness(
            'n_channels', known_type=int, choices=[1, 2])

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['train', 'test', 'val',
                                              'train_train'])

        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)
        self.augment = True

        self.n_samples = self.get_arg_and_check_validness(
            'n_samples', known_type=int, extra_lambda_checks=[lambda x: x >= 0])

        self.sample_rate = self.get_arg_and_check_validness(
            'sample_rate', known_type=int, choices=[8000, 44100])

        self.root_path = self.get_arg_and_check_validness(
            'root_dirpath', known_type=str,
            choices=[MUSDBWAV_ROOT_PATH, MUSDBWAV8K_ROOT_PATH, MUSDB_ROOT_PATH],
            extra_lambda_checks=[lambda y: os.path.lexists(y)])

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.time_samples = int(self.sample_rate * self.timelength)

        # Fetch the dataloader
        if self.root_path == MUSDB_ROOT_PATH:
            wav_version = False
        else:
            wav_version = True

        if self.split in ['train', 'test']:
            self.data_access = musdb_lib.DB(
                subsets=self.split, root=self.root_path, is_wav=wav_version)
        elif self.split == 'val':
            self.data_access = musdb_lib.DB(subsets='train', split='val',
                                            root=self.root_path,
                                            is_wav=wav_version)
        else:
            self.data_access = musdb_lib.DB(subsets='train', split='train',
                                            root=self.root_path,
                                            is_wav=wav_version)

        for track in self.data_access:
            assert float(track.rate) == self.sample_rate, 'Mismatched sample rate.'

        self.n_tracks = len(self.data_access)
        self.source_types = ['drums', 'bass', 'other', 'vocals']

        # If it's not an augmented dataset, that means we need to go through
        # all the songs.
        self.predefined_indexes = None
        if not self.augment:
            self.predefined_indexes = []
            print('Fixing the predefined indices for the dataset...')
            for i, track in enumerate(self.data_access):
                if self.timelength < 0:
                    self.predefined_indexes.append(
                        {'track': i, 'st_time': 0.,
                         'chunk_duration': track.duration})
                else:
                    n_start_indices = int(track.duration / self.timelength)
                    the_st_ind = np.random.choice(n_start_indices)
                    self.predefined_indexes.append(
                        {'track': i, 'st_time': the_st_ind * self.timelength,
                         'chunk_duration': self.timelength})
                    # for j in range(n_start_indices):
                    #     self.predefined_indexes.append(
                    #         {'track': i, 'st_time': j * self.timelength,
                    #          'chunk_duration': self.timelength})
                    # self.predefined_indexes.append(
                    #     {'track': i, 'st_time': 0.,
                    #      'chunk_duration': self.timelength})

            self.n_samples = len(self.predefined_indexes)

    def __len__(self):
        return self.n_samples

    def _get_track_segment_data(self, idx):
        if not self.augment:
            # Get the predefined data
            selected_example = self.predefined_indexes[idx]
            track_id = selected_example['track']
            selected_track = self.data_access.tracks[track_id]
            st_time = selected_example['st_time']
            chunk_duration = selected_example['chunk_duration']
        else:
            # Draw some random segment
            the_time = int(np.modf(time())[0] * 100000000)
            np.random.seed(the_time)

            track_id = np.random.choice(self.n_tracks)
            selected_track = self.data_access.tracks[track_id]
            if self.timelength < 0:
                st_time = 0.
                chunk_duration = selected_track.duration
            else:
                st_time = np.random.uniform(
                    0., selected_track.duration - self.timelength)
                chunk_duration = self.timelength
        selected_track.chunk_duration = chunk_duration
        selected_track.chunk_start = st_time
        return selected_track.stems

    def safe_pad(self, tensor_wav):
        if self.zero_pad and tensor_wav.shape[-1] < self.time_samples:
            appropriate_shape = tensor_wav.shape
            padded_wav = torch.zeros(
                list(appropriate_shape[:-1]) + [self.time_samples],
                dtype=torch.float32)
            padded_wav[..., :tensor_wav.shape[0]] = tensor_wav
            return padded_wav[..., :self.time_samples]
        else:
            return tensor_wav

    def __getitem__(self, idx):
        data = self._get_track_segment_data(idx)
        # data has shape: (5, time_samples, 2)
        # The sources in order are: ['mix', 'drums', 'bass', 'other', 'vocals']
        data = torch.tensor(data, dtype=torch.float32)
        data = data.permute(2, 0, 1)
        data = self.safe_pad(data)

        # Return only the clean sources for mono or stereo
        if self.n_channels == 1:
            return torch.sum(data[:, 1:, :], 0, keepdim=True)
        else:
            return data[:, 1:, :]

    def get_generator(self, batch_size=4, shuffle=True, num_workers=4):
        generator_params = {'batch_size': batch_size,
                            'shuffle': shuffle,
                            'num_workers': num_workers,
                            'drop_last': True}
        return torch.utils.data.DataLoader(self, **generator_params)


def test_generator():
    batch_size = 3
    sample_rate = 8000
    timelength = 4.0

    time_samples = int(sample_rate * timelength)
    for n_channels in [1, 2]:
        data_loader = Dataset(
            root_dirpath=MUSDBWAV8K_ROOT_PATH, n_channels=n_channels,
            augment=True,
            split='train', sample_rate=sample_rate, timelength=timelength,
            zero_pad=True, normalize_audio=False, n_samples=10)
        generator = data_loader.get_generator(batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

        for sources in generator:
            assert sources.shape == (batch_size, n_channels, 4, time_samples)


    # # test the testing set with batch size 1 only
    # data_loader = Dataset(
    #     root_dirpath=MUSDBWAV8K_ROOT_PATH, n_channels=1, augment=False,
    #     split='test', sample_rate=sample_rate, timelength=-1.,
    #     zero_pad=False, normalize_audio=False, n_samples=10)
    # generator = data_loader.get_generator(batch_size=1, num_workers=2,
    #                                       shuffle=False)
    #
    # direct_data_access = musdb_lib.DB(subsets='test',
    #                                   root=MUSDBWAV8K_ROOT_PATH, is_wav=True)
    #
    # for i, sources in enumerate(generator):
    #     assert sources.shape[:-1] == (1, 1, 4)
    #     assert np.isclose(sources.shape[-1] / sample_rate,
    #                       direct_data_access.tracks[i].duration)



if __name__ == "__main__":
    test_generator()