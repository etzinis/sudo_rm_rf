"""!
@brief A dataset creation which is compatible with pytorch framework
and much faster in loading time depending on the new version of
loading only the appropriate files that might be needed. Moreover
this dataset has minimal input argument requirements in order to be
more user friendly.

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

current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)

from torch.utils.data import Dataset, DataLoader
from sudo_rm_rf.utils import preprocess_wsj0mix


class End2EndMixtureDataset(Dataset):
    """
    This is a general compatible class for pytorch datasets.

    @note Each instance of the dataset should be stored using
    joblib.dump() and this is the way that it would be returned.
    After some transformations.

    The path of all datasets should be defined inside config.
    All datasets should be formatted with appropriate subfolders of
    train / test and val and under them there should be all the
    available files.
    """
    def __init__(self,
                 **kwargs):
        """!
        In this initialization pyto rch dataloader function there is
        some checking for all the available arguments.

        \warning
         Input dataset dir should have the following structure:
         ./dataset_dir
             ./train
                ./uid
                    mixture_wav
                    clean_wavs
             ./test
             ./val


        """
        self.kwargs = kwargs

        self.dataset_dirpath = self.get_arg_and_check_validness(
                               'input_dataset_p',
                               known_type=str,
                               extra_lambda_checks=
                               [lambda x: os.path.lexists(x)])

        self.dataset_samples_folders = glob2.glob(os.path.join(
            self.dataset_dirpath, '*'))

        self.n_items = len(self.dataset_samples_folders)

        self.get_top = self.get_arg_and_check_validness('get_top')
        if isinstance(self.get_top, int):
            self.n_items = min(self.n_items, self.get_top)
            self.dataset_samples_folders = \
                self.dataset_samples_folders[:self.n_items]

        self.n_jobs = self.get_arg_and_check_validness(
                      'n_jobs',
                      known_type=int,
                      extra_lambda_checks=
                      [lambda x: x <= psutil.cpu_count()])

        self.batch_size = self.get_arg_and_check_validness(
            'batch_size',
            known_type=int,
            extra_lambda_checks=
            [lambda x: x <= self.n_items])

        # The following name is considered:
        # ['bpd' -> Binary Phase Difference Mask
        #  'ds' -> Binary Dominating Source Mask
        #  'rpd' -> Continuous Raw Phase Difference Mask]
        #  'mixture_wav' -> The wav of the mixture on mic 1 (1d numpy)
        #  'bpd_sources_wavs' -> Wavs for K sources reconstructed by
        # applying bpd mask and istft (2d (K rows) numpy)
        #  recorded on mic 1
        #  'clean_sources_wavs'-> Wavs for K clean sources
        # (2d (K rows) numpy) recorded on mic 1
        self.return_items = self.get_arg_and_check_validness(
            'return_items',
            known_type=list,
            choices=['mixture_wav',
                     'clean_sources_wavs',
                     'mic1_wav_downsampled',
                     'clean_sources_wavs_downsampled',
                     'mixture_wav_norm',
                     'clean_sources_wavs_norm'])

        self.n_batches = int(self.n_items / self.batch_size)

        self.n_sources = self.infer_n_sources()

    def get_n_batches(self):
        return self.n_batches

    def infer_n_sources(self):
        try:
            name = os.path.basename(os.path.dirname(
                self.dataset_dirpath.strip('/')))
            if 'wsj' in name:
                _, n_sources, _, _ = preprocess_wsj0mix.parse_info_from_name(
                    name)
            elif 'timit' in name:
                n_sources = int(name.split("_")[4])
            else:
                raise IOError
        except:
            raise(IOError("Cannot infer the number of sources in the "
                          "mixture dataset from naming! Warning it is"
                          " expected to get the number of sources as "
                          "the 4th integer on the naming of the "
                          "input dataset: e.g. "
                          "timit_5400_1800_512_2_fm_random_taus_delays"
                          "means 2 sources of female + male mixtures"))
        return n_sources

    def get_n_sources(self):
        return self.n_sources

    def __len__(self):
        return self.n_items

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

    def __getitem__(self, idx):
        """!
        Depending on the selected partition it returns accordingly
        the following objects:

        depending on the list of return items the caller function
        will be returned the items in the exact same order

        @throws If one of the desired objects cannot be loaded from
        disk then an IOError would be raised
        """

        items_folder = self.dataset_samples_folders[idx]

        return [self.load_item_file(os.path.join(items_folder, name))
                for name in self.return_items]


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='Pytorch Dataset Loader for Raw Wavs and/or Masks '
                    'on the Fourie domain')
    parser.add_argument("-i", "--input_dataset_p", type=str,
                        help="Dataset path you want to load",
                        default=None, required=True)
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                            Warning: Cannot be less than the number of 
                            the validation samples""", default=3)
    parser.add_argument("--n_jobs", type=int,
                        help="""The number of cpu workers for 
                            loading the data, etc.""", default=4)
    parser.add_argument("--get_top", type=int,
                        help="""Reduce the number of this dataset 
                                samples to this number.""",
                        default=None)
    parser.add_argument("-ri", "--return_items", type=str, nargs='+',
                        help="""A list of elements that this 
                        dataloader should return. See available 
                        choices which are based on the saved data 
                        names which are available. There is no type 
                        checking in this return argument.""",
                        default=['mixture_wav',
                                 'clean_sources_wavs'],
                        required=True,
                        choices=['mixture_wav',
                                 'clean_sources_wavs',
                                 'mic1_wav_downsampled',
                                 'clean_sources_wavs_downsampled',
                                 'mixture_wav_norm',
                                 'clean_sources_wavs_norm'])
    return parser.parse_args()


def get_data_gen_from_loader(data_loader):

    generator_params = {'batch_size': data_loader.batch_size,
                        'shuffle': True,
                        'num_workers': data_loader.n_jobs,
                        'drop_last': True}
    data_generator = DataLoader(data_loader,
                                **generator_params)
    return data_generator


def get_data_generators(
    data_paths=[''],
    bs=16,
    n_jobs=3,
    get_top=[None],
    return_items=['mixture_wav',
                  'clean_sources_wavs']):
    assert len(get_top) == len(data_paths)
    generators = []

    for path, n_elements in zip(data_paths, get_top):
        args = argparse.Namespace(
            input_dataset_p=path,
            batch_size=bs,
            n_jobs=n_jobs,
            get_top=n_elements,
            return_items=return_items
        )
        subset_DS = End2EndMixtureDataset(**vars(args))
        subset_gen = get_data_gen_from_loader(subset_DS)
        generators.append(subset_gen)

    return generators


def example_of_usage(pytorch_dataloader_args):
    """!
    Simple example of how to use this pytorch data loader"""
    # lets change the list of the return items:
    # pytorch_dataloader_args.return_items = ['mixture_wav',
    #                                         'clean_sources_wavs']

    data_loader = End2EndMixtureDataset(**vars(pytorch_dataloader_args))
    data_gen = get_data_gen_from_loader(data_loader)

    batch_cnt = 0
    print("Loading {} Batches of size: {} for mixtures with {} active "
          "sources...".format(
          data_loader.get_n_batches(),
          data_loader.batch_size,
          data_loader.get_n_sources()) + "\n" + "=" * 20 + "\n")

    for batch_data_list in data_gen:
        # the returned elements are tensors
        # Always the first dimension is the selected batch size
        mixture_wav, bpd_sources_wavs = batch_data_list
        if not batch_cnt:
            print("Returned mixture_wav of type: {} and size: "
                  "{}".format(type(mixture_wav),
                              mixture_wav.size()))
            print("Returned bpd_sources_wavs of type: {} and size: "
                  "{}".format(type(bpd_sources_wavs),
                              bpd_sources_wavs.size()))
            batch_cnt += 1


if __name__ == "__main__":
    pytorch_dataloader_args = get_args()
    example_of_usage(pytorch_dataloader_args)
