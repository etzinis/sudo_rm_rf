"""!
@brief Pytorch dataloader for online mixing of sources audio of multiple
hierarchical datasets.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import argparse
import os
import glob2
import psutil
import inspect
import joblib
import sys
import numpy as np
import torch
from time import time

current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)

from torch.utils.data import Dataset, DataLoader
from __config__ import ESC50_DOWNLOADED_P, ESC50_HIERARCHICAL_P, WSJ_MIX_HIERARCHICAL_P


class AugmentedOnlineMixingDataset(Dataset):
    """
    This is a general compatible class for pytorch datasets with hierarchical
    structure. That means that for each dataset which is given the following
    directory tree structure is assumed:

    dataset1/
        ...
        class_sound_1/
            ...
            sample_x/
                ...
                torch_tensor_to_load
                ...
            ...
        ...

    @note Each instance of the dataset should be stored using
    joblib.dump() and this is the way that it would be returned.

    The path of all datasets should be defined inside config.
    All datasets should be formatted with appropriate subfolders of
    train / test and val and under them there should be all the
    available files.
    """
    def __init__(self,
                 **kwargs):
        """!
        The user can also specify whether there should be a specific prior
        probability of finding samples from one dataset in the mixtures or to
        have a fixed dataset (useful for evaluation or test partitions).
        """
        self.kwargs = kwargs

        self.datasets_dirpaths = self.get_arg_and_check_validness(
            'input_dataset_p',
            known_type=list,
            extra_lambda_checks=
            [lambda y: all([os.path.lexists(x) for x in y])])

        self.n_datasets = len(self.datasets_dirpaths)

        self.datasets_priors = self.get_arg_and_check_validness(
            'datasets_priors',
            known_type=list,
            extra_lambda_checks=[
                lambda x: len(x) == len(self.datasets_dirpaths),
                lambda y: sum(y) == 1.])
        self.priors_cdf = np.cumsum(np.array(self.datasets_priors))

        self.n_samples = self.get_arg_and_check_validness(
            'n_samples',
            known_type=int,
            extra_lambda_checks=[lambda x: x > 0])

        self.fs = self.get_arg_and_check_validness('fs', known_type=float)

        self.selected_timelength = self.get_arg_and_check_validness(
            'selected_timelength', known_type=float)
        self.selected_wav_samples = int(self.fs * self.selected_timelength)

        self.max_abs_snr = self.get_arg_and_check_validness(
            'max_abs_snr',
            known_type=float,
            extra_lambda_checks=[lambda x: x > 0])

        self.n_sources = self.get_arg_and_check_validness(
            'n_sources',
            known_type=int,
            extra_lambda_checks=[lambda x: x == 2])

        self.n_jobs = self.get_arg_and_check_validness(
                      'n_jobs',
                      known_type=int,
                      extra_lambda_checks=
                      [lambda x: x <= psutil.cpu_count()])

        self.batch_size = self.get_arg_and_check_validness(
            'batch_size',
            known_type=int,
            extra_lambda_checks=
            [lambda x: x <= self.n_samples])

        # First will be returned the mixture and the source
        self.return_items = self.get_arg_and_check_validness(
            'return_items',
            known_type=list,
            choices=['wav', 'fold', 'class_id',
                     'human_readable_class', 'source_file'],
            extra_lambda_checks=
            [lambda x: x[0] == 'wav' or x[0] == 'wav_norm',
             lambda x: not('wav' in x and 'wav_norm' in x)]
        )

        self.n_batches = int(self.n_samples / self.batch_size)

        # Create the list of lists representation of the whole dataset.
        # In order to index this list of lists:
        # self.data[dataset_idx][hierarchical_folder_idx][sample_idx]
        self.hierarchical_folders = [
            glob2.glob(dp + '/*') for dp in self.datasets_dirpaths]
        self.n_hierarchical_folders = [
            len(dataset_folders)
            for dataset_folders in self.hierarchical_folders
        ]

        self.sample_folders = []
        self.n_sample_folders = []

        for dataset_folders in self.hierarchical_folders:

            hier_folders_samples = []
            n_hier_folders_samples = []
            for hierachical_folder in dataset_folders:
                these_samples = glob2.glob(hierachical_folder + '/*')
                hier_folders_samples.append(these_samples)
                n_hier_folders_samples.append(len(these_samples))
            self.sample_folders.append(hier_folders_samples)
            self.n_sample_folders.append(n_hier_folders_samples)

        # If the dataset is fixed then just create the whole indexing
        # beforehand.
        self.fixed_seed = self.get_arg_and_check_validness(
            'fixed_seed', known_type=int, extra_lambda_checks=[lambda x: x >= 0])
        if self.fixed_seed == 0:
            print('Dataset is going to be created online for: {} '
                  'samples'.format(self.n_samples))
            self.random_draws = None
        else:
            print('Dataset is fixed for: {} samples'.format(self.n_samples))
            np.random.seed(self.fixed_seed)
            self.random_draws = np.random.random(
                (self.n_samples, self.n_sources, 5))

    def get_n_batches(self):
        return self.n_batches

    def __len__(self):
        return self.n_samples

    def get_arg_and_check_validness(self,
                                    key,
                                    choices=None,
                                    known_type=None,
                                    extra_lambda_checks=None):

        try:
            value = self.kwargs[key]
        except:
            raise KeyError("Argument: <{}> does not exist in pytorch "
                           "dataloader keyword arguments".format(key))

        if known_type is not None:
            if not isinstance(value, known_type):
                raise TypeError("Value: <{}> for key: <{}> is not an "
                                "instance of "
                                "the known selected type: <{}>"
                                "".format(value, key, known_type))

        if choices is not None:
            if isinstance(value, list):
                if not all([v in choices for v in value]):
                    raise ValueError("Values: <{}> for key: <{}>  "
                                     "contain elements in a"
                                     "regime of non appropriate "
                                     "choices instead of: <{}>"
                                     "".format(value, key, choices))
            else:
                if value not in choices:
                    raise ValueError("Value: <{}> for key: <{}> is "
                                     "not in the "
                                     "regime of the appropriate "
                                     "choices: <{}>"
                                     "".format(value, key, choices))

        if extra_lambda_checks is not None:
            all_checks_passed = all([f(value)
                                     for f in extra_lambda_checks])
            if not all_checks_passed:
                raise ValueError("Value(s): <{}> for key: <{}>  "
                "does/do not fulfil the predefined checks: <{}>".format(
                    value, key,
                    [inspect.getsourcelines(c)[0][0].strip()
                     for c in extra_lambda_checks
                     if not c(value)]))

        return value

    @staticmethod
    def load_item_file(path):
        try:
            loaded_file = joblib.load(path)
        except:
            raise IOError("Failed to load data file from path: {} "
                          "".format(path))
        return loaded_file

    def get_selected_dataset_index(self, sample_idx, source_idx):
        if self.random_draws is None:
            random_draw = np.random.random()
        else:
            random_draw = self.random_draws[sample_idx, source_idx, 0]

        for dataset_idx in range(self.n_datasets):
            if random_draw < self.priors_cdf[dataset_idx]:
                return dataset_idx
        return self.n_datasets - 1

    def get_selected_hierarchical_folder_index(
            self, sample_idx, source_idx, dataset_idx, not_equal_to=None):
        if self.random_draws is None:
            random_draw = np.random.random()
        else:
            random_draw = self.random_draws[sample_idx, source_idx, 1]

        ind = int(random_draw * self.n_hierarchical_folders[dataset_idx])
        if not_equal_to is not None:
            if ind == not_equal_to:
                ind = (ind + 1) % self.n_hierarchical_folders[dataset_idx]

        return ind

    def get_selected_sample_folder_index(
            self, sample_idx, source_idx, dataset_idx, hierarchical_folder_idx):
        if self.random_draws is None:
            random_draw = np.random.random()
        else:
            random_draw = self.random_draws[sample_idx, source_idx, 2]

        return int(random_draw *
                   self.n_sample_folders[dataset_idx][hierarchical_folder_idx])

    def get_sample_delay(self, sample_idx, source_idx, tensor_samples):
        if self.random_draws is None:
            random_draw = np.random.random()
        else:
            random_draw = self.random_draws[sample_idx, source_idx, 3]

        return int(random_draw * (tensor_samples - self.selected_wav_samples))

    def get_snr_ratio(self, sample_idx, source_idx):
        if self.random_draws is None:
            random_draw = np.random.random()
        else:
            random_draw = self.random_draws[sample_idx, source_idx, 4]

        return (random_draw - 0.5) * self.max_abs_snr * 2
        # return np.random.normal(2.507, 2.1)

    def __getitem__(self, mixture_idx):
        """!
        Depending on the selected partition it returns accordingly
        the following objects:

        depending on the list of return items the caller function
        will be returned the items in the exact same order

        @throws If one of the desired objects cannot be loaded from
        disk then an IOError would be raised
        """
        if self.random_draws is None:
            the_time = int(np.modf(time())[0] * 100000000)
            np.random.seed(the_time)

        sources_wavs_l = []
        energies = []
        prev_indexes = []
        extra_files = []

        # Select with a prior probability between the list of datasets
        for source_idx in range(self.n_sources):
            dataset_idx = self.get_selected_dataset_index(
                mixture_idx, source_idx)

            # Avoid getting the same sound class inside the mixture
            not_equal_to = None
            if len(prev_indexes) > 0:
                prev_d_ind, prev_h_ind = prev_indexes[0]
                if prev_d_ind == dataset_idx:
                    not_equal_to = prev_h_ind
            hier_folder_idx = self.get_selected_hierarchical_folder_index(
                mixture_idx, source_idx, dataset_idx, not_equal_to=not_equal_to)
            wav_idx = self.get_selected_sample_folder_index(
                mixture_idx, source_idx, dataset_idx, hier_folder_idx)

            prev_indexes.append([dataset_idx, hier_folder_idx])
            item_folder = self.sample_folders[dataset_idx][hier_folder_idx][wav_idx]
            source_tensor = self.load_item_file(os.path.join(
                item_folder, self.return_items[0]))

            # Random shifting of the source signals
            samples_delay = self.get_sample_delay(
                mixture_idx, source_idx, source_tensor.shape[-1])
            delayed_source_tensor = source_tensor[
                :, samples_delay:samples_delay+self.selected_wav_samples]

            if np.allclose(delayed_source_tensor, 0):
                delayed_source_tensor = source_tensor[:, :self.selected_wav_samples]

            # Random SNR mixing
            energies.append(torch.sqrt(torch.sum(delayed_source_tensor ** 2)))
            sources_wavs_l.append(delayed_source_tensor)

            if len(self.return_items) > 1:
                if len(extra_files) == 0:
                    extra_files = [
                        [self.load_item_file(os.path.join(item_folder, file))]
                        for file in self.return_items[1:]]
                else:
                    for j, file in enumerate(self.return_items[1:]):
                        extra_files[j].append(
                            self.load_item_file(os.path.join(item_folder, file)))

        snr_ratio = self.get_snr_ratio(mixture_idx, 0)
        new_energy_ratio = np.sqrt(np.power(10., snr_ratio / 10.))

        sources_wavs_l[0] = new_energy_ratio * sources_wavs_l[0] / (
            energies[0] + 10e-8)
        sources_wavs_l[1] = sources_wavs_l[1] / (energies[1] + 10e-8)

        clean_sources_tensor = torch.cat(sources_wavs_l)
        mixture_tensor = torch.sum(clean_sources_tensor, dim=0, keepdim=True)

        clean_sources_tensor -= torch.mean(clean_sources_tensor, dim=1,
                                           keepdim=True)
        mixture_tensor -= torch.mean(mixture_tensor, dim=1, keepdim=True)
        mixture_std = torch.std(mixture_tensor, dim=1)

        returning_mixture = (mixture_tensor / (mixture_std + 10e-8)).squeeze()
        returning_sources = clean_sources_tensor / (mixture_std + 10e-8)

        if len(self.return_items) > 1:
            # preprocess exta files format for tensor data
            for j, info in enumerate(extra_files):
                if type(info[0]) == torch.Tensor:
                    extra_files[j] = torch.stack(info)
            return [returning_mixture, returning_sources] + extra_files
        else:
            return returning_mixture, returning_sources


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='Augmented online mixing and loading')
    parser.add_argument("-i", "--input_dataset_p", type=str, nargs='+',
                        help="Hierarchical Dataset paths you want to load from",
                        default=None, required=True)
    parser.add_argument("-priors", "--datasets_priors", type=float, nargs='+',
                        help="The prior probability of finding a sample from "
                             "each given dataset. The length of this list "
                             "must be equal to the number of dataset paths "
                             "given above. The sum of this list must add up "
                             "to 1.",
                        default=None, required=True)
    parser.add_argument("-fs", type=float,
                        help="""Sampling rate of the audio.""", default=8000.)
    parser.add_argument("--selected_timelength", type=float,
                        help="""The timelength of the sources that you want 
                            to load in seconds.""",
                        default=4.)
    parser.add_argument("--max_abs_snr", type=float,
                        help="""The maximum absolute value of the SNR of 
                            the mixtures.""", default=2.5)
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                            Warning: Cannot be less than the number of 
                            the validation samples""", default=3)
    parser.add_argument("--n_sources", type=int,
                        help="""The number of sources inside each mixture 
                        which is generated""",
                        default=2)
    parser.add_argument("--n_jobs", type=int,
                        help="""The number of cpu workers for 
                            loading the data, etc.""", default=4)
    parser.add_argument("--fixed_seed", type=int,
                        help="""Whether the dataset is going to be fixed (
                        e.g. test partitions should be always fixed) give 
                        the random seed. If seed is zero then it means that 
                        the dataset is not going to be fixed.""",
                        default=0)
    parser.add_argument("--n_samples", type=int,
                        help="""Define the number of this dataset samples.""",
                        required=True)
    parser.add_argument("-ri", "--return_items", type=str, nargs='+',
                        help="""A list of elements that this 
                        dataloader should return. See available 
                        choices which are based on the saved data 
                        names which are available. There is no type 
                        checking in this return argument.""",
                        default=['wav'],
                        choices=['wav', 'wav_norm'])
    return parser.parse_args()


def get_data_gen_from_loader(data_loader):

    generator_params = {'batch_size': data_loader.batch_size,
                        'shuffle': True,
                        'num_workers': data_loader.n_jobs,
                        'drop_last': True}
    data_generator = DataLoader(data_loader,
                                **generator_params)
    return data_generator


def example_of_usage(pytorch_dataloader_args):
    """!
    Simple example of how to use this pytorch data loader"""

    data_loader = AugmentedOnlineMixingDataset(**vars(pytorch_dataloader_args))
    data_gen = get_data_gen_from_loader(data_loader)

    batch_cnt = 0
    print("Loading {} Batches of size: {} for mixtures with {} active "
          "sources...".format(
          data_loader.get_n_batches(),
          data_loader.batch_size,
          data_loader.n_sources) + "\n" + "=" * 20 + "\n")

    from tqdm import tqdm
    for batch_data_list in tqdm(data_gen):
        # the returned elements are tensors
        # Always the first dimension is the selected batch size
        mixture_wav, sources_wavs = batch_data_list
        if not batch_cnt:
            print("Returned mixture_wav of type: {} and size: "
                  "{}".format(type(mixture_wav),
                              mixture_wav.size()))
            print("Returned sources_wavs of type: {} and size: "
                  "{}".format(type(sources_wavs),
                              sources_wavs.size()))
            batch_cnt += 1


def test_truly_random_generator():
    these_args = argparse.Namespace(
        input_dataset_p=[os.path.join(WSJ_MIX_HIERARCHICAL_P, 'train'),
                         os.path.join(ESC50_HIERARCHICAL_P, 'train')],
        datasets_priors=[0.5, 0.5],
        batch_size=1,
        n_jobs=4,
        n_samples=2,
        return_items=['wav'],
        fs=8000.,
        selected_timelength=4.,
        n_sources=2,
        max_abs_snr=2.5,
        fixed_seed=0
    )

    data_loader = AugmentedOnlineMixingDataset(**vars(these_args))
    gen = get_data_gen_from_loader(data_loader)

    for mixture, sources,  in gen:
        prev_mix_wav = mixture.squeeze()
        prev_s1_wav = sources.squeeze()[0]
        prev_s2_wav = sources.squeeze()[1]

    for i in range(500):
        for mixture, sources in gen:
            mix_wav = mixture.squeeze()
            s1_wav = sources.squeeze()[0]
            s2_wav = sources.squeeze()[1]
            if (torch.allclose(mix_wav, prev_mix_wav) or
                torch.allclose(s1_wav, prev_s1_wav) or
                torch.allclose(s2_wav, prev_s2_wav)):
                raise ValueError('Dataset generator is not truly random')


def test_metadata_loading():
    bs, n_samples, fs, timelength, n_sources = 2, 4000, 8000., 3.096, 2
    these_args = argparse.Namespace(
        input_dataset_p=[os.path.join(WSJ_MIX_HIERARCHICAL_P, 'train')],
        datasets_priors=[1.],
        batch_size=bs,
        n_jobs=4,
        n_samples=n_samples,
        return_items=['wav'],
        # return_items=['wav', 'fold', 'class_id',
        #               'human_readable_class', 'source_file'],
        fs=fs,
        selected_timelength=timelength,
        n_sources=n_sources,
        max_abs_snr=2.5,
        fixed_seed=0
    )

    data_loader = AugmentedOnlineMixingDataset(**vars(these_args))
    gen = get_data_gen_from_loader(data_loader)

    for data in gen:
        # for i in range(4):
        #     assert type(data[i]) == torch.Tensor
        assert data[0].shape == torch.Size([int(bs),
                                            int(fs * timelength)])
        assert data[1].shape == torch.Size([int(bs), int(n_sources),
                                            int(fs * timelength)])
        # assert data[2].shape == torch.Size([int(bs), int(n_sources)])
        # assert data[3].shape == torch.Size([int(bs), int(n_sources)])


if __name__ == "__main__":
    # pytorch_dataloader_args = get_args()
    # example_of_usage(pytorch_dataloader_args)
    # test_truly_random_generator()
    test_metadata_loading()