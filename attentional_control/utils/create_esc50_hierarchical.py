"""!
@brief Data Preprocessor for esc-50 in order to convert the dataset to an
hierarchical one for better loadin from a pytorch data loader.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(root_dir)

from glob2 import glob
from scipy.io.wavfile import read as scipy_wavread
import joblib
import librosa
from scipy.io import wavfile
import torch
from tqdm import tqdm
import shutil

import attentional_control.utils.progress_display as progress_display

import time
import numpy as np
import matplotlib.pyplot as plt


def parse_info_from_name(preprocessed_dirname):
    """! Given a name of a preprocessed dataset root dirname infer all the
    encoded information

    Args:
        preprocessed_dirname: A dirname as the one follows:
        wsj0_2mix_8k_4s_min_preprocessed or even a dirpath

    Returns:
        min_or_max: String whether the mixtures are aligned with the maximum
        or the minimum number of samples of the constituting sources
        n_speakers: number of speakers in mixtures
        fs: sampling rate in kHz
        wav_timelength: The timelength in seconds of all the mixtures and
                        clean sources
    """
    try:
        dirname = os.path.basename(preprocessed_dirname)
        elements = dirname.split('_')
        min_or_max = elements[-2]
        assert (min_or_max == 'min' or min_or_max == 'max')
        wav_timelength = float(elements[-3][:-1])
        fs = float(elements[-4][:-1]) * 1000
        n_speakers = int(elements[-5].strip("mix"))

        return min_or_max, n_speakers, fs, wav_timelength

    except:
        raise IOError("The structure of the wsj0-mix preprocessed "
                      "dataset name is not in the proper format. A proper "
                      "format would be: "
                      "wsj0_{number of speakers}mix_{fs}k_{timelength}s_{min "
                      "or max}_preprocessed")


def infer_output_name(input_dirpath, wav_timelength):
    """! Infer the name for the output folder as shown in the example: E.g.
    for input_dirpath: wsj0-mix/2speakers/wav8k/min and for 4s timelength it
    would be wsj0_2mix_8k_4s_min_preprocessed

    Args: input_dirpath: The path of a wsj0mix dataset e.g.
                         wsj0-mix/2speakers/wav8k/min (for mixes with minimum
                         length)
          wav_timelength: The timelength in seconds of all the mixtures and
                          clean sources

    Returns: outputname: as specified in string format
             fs: sampling rate in Hz
             n_speakers: number of speakers in mixtures
    """

    try:
        elements = input_dirpath.lower().split('/')
        min_or_max = elements[-1]
        assert (min_or_max == 'min' or min_or_max == 'max')
        fs = int(elements[-2].strip('wav').strip('k')) * 1000
        n_speakers = int(elements[-3].strip("speakers"))
        output_name = "wsj0_{}mix_{}k_{}s_{}_hierarchical".format(
            n_speakers,
            fs / 1000.,
            float(wav_timelength),
            min_or_max)

        # test that the inferred output name is parsable back
        (inf_min_or_max,
         inf_n_speakers,
         inf_fs,
         inf_wav_timelength) = parse_info_from_name(output_name)
        assert(inf_min_or_max == min_or_max and
               inf_n_speakers == n_speakers and
               inf_fs == fs and
               inf_wav_timelength == wav_timelength)

        return output_name, fs, n_speakers, min_or_max

    except:
        raise IOError("The structure of the wsj0-mix is not in the right "
                "format. A proper format would be: "
                "wsj0-mix/{2 or 3}speakers/wav{fs in Hz}k/{min or max}")


def normalize_wav(wav, eps=10e-7, std=None):
    mean = wav.mean()
    if std is None:
        std = wav.std()
    return (wav - mean) / (std + eps)


def write_data_wrapper_func(audio_files_dir,
                            sound_classes,
                            min_wav_timelength,
                            output_dirpath):
    min_wav_samples = int(8000 * min_wav_timelength)

    def process_uid(uid):
        audio_path = os.path.join(audio_files_dir, uid)
        sound_class = uid.split('-')[-1].split('.wav')[0]
        if sound_class not in sound_classes:
            raise ValueError('Sound Class: {} must be in Classes: {}'.format(
                sound_class, sound_classes))
        y, sr = librosa.load(audio_path, sr=44100)
        y_8k = librosa.resample(y, sr, 8000)

        if y_8k.shape[0] < min_wav_samples:
            return
        wav = torch.tensor(y_8k, dtype=torch.float32).unsqueeze(0)
        norm_wav = torch.tensor(normalize_wav(y_8k),
                                dtype=torch.float32).unsqueeze(0)
        output_uid_folder = os.path.join(
            output_dirpath, sound_class, uid.split('.wav')[0])

        data = {
            'wav': wav,
            'wav_norm': norm_wav,
        }

        if not os.path.exists(output_uid_folder):
            os.makedirs(output_uid_folder)

        for k, v in data.items():
            file_path = os.path.join(output_uid_folder, k)
            joblib.dump(v, file_path, compress=0)

    return lambda uid: process_uid(uid)


def convert_ESC50_to_hierarchical_dataset(input_dirpath,
                                          output_dirpath,
                                          wav_timelength):
    """! This function converts the ESC-50 dataset in a universal
    dataset with all different sound classes in separate folders.

    Args:
        input_dirpath: The path of a ESC-50 dataset
        output_dirpath: The path for storing the new dataset
                        (the directories would be created recursively)
        wav_timelength: The timelength in for the clean sources to be longer
                        than

    Intermediate:
        output_name: Default name would be infered as follows:
                     E.g. for wsj0-mix/2speakers/wav8k/min and for 4s
                     timelength:
                     it would be wsj0_2mix_8k_4s_min_hierarchical
    """

    audio_files_dir = os.path.join(input_dirpath, 'audio')
    files = glob(audio_files_dir + '/*.wav')
    unique_ids = set([os.path.basename(f) for f in files])

    sound_classes = set([u.split('-')[-1].split('.wav')[0]
                         for u in unique_ids])

    write_data_func = write_data_wrapper_func(audio_files_dir,
                                              sound_classes,
                                              wav_timelength,
                                              output_dirpath)

    progress_display.progress_bar_wrapper(
        write_data_func,
        list(unique_ids),
        message='Processing {} files...'.format(len(unique_ids)))

    print("Dataset ready at: {}".format(output_dirpath))


def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)


def partition_dataset(hier_dataset_dirpath,
                      partitions_dirpath):
    # After converting the dataset to an hierarchical one then partition it
    # to train val and test. The folds are going to be used as train
    # partition is going to get folds 1-4 for all sound classes while test
    # and val are going to share the remaining 5th fold
    sound_classes_dirs = glob(hier_dataset_dirpath + '/*')
    for class_path in tqdm(sound_classes_dirs):
        class_name = os.path.basename(class_path)
        audio_files_dirs = glob(class_path + '/*')

        samples_dict = dict([(x, []) for x in range(1, 6)])
        data_sample_names = [os.path.basename(file_dir)
                             for file_dir in audio_files_dirs]
        for data_sample_name in data_sample_names:
            fold = int(data_sample_name.split('-')[0])
            samples_dict[fold].append(data_sample_name)

        folds_numbers = sorted(samples_dict.keys())
        for k in folds_numbers[:-1]:
            for sample in samples_dict[k]:
                out_dir = os.path.join(partitions_dirpath, 'train', class_name,
                                       sample)
                copyDirectory(os.path.join(class_path, sample), out_dir)

        n_val_test_samples = len(samples_dict[folds_numbers[-1]])
        n_test_samples = int(n_val_test_samples / 2.)
        for sample in samples_dict[folds_numbers[-1]][:n_test_samples]:
            out_dir = os.path.join(partitions_dirpath, 'test', class_name, sample)
            copyDirectory(os.path.join(class_path, sample), out_dir)

        for sample in samples_dict[folds_numbers[-1]][n_test_samples:]:
            out_dir = os.path.join(partitions_dirpath, 'val', class_name, sample)
            copyDirectory(os.path.join(class_path, sample), out_dir)


def example_of_usage():
    input_dirpath = '/mnt/data/hierarchical_sound_datasets/ESC-50-master'
    output_dirpath = '/mnt/data/hierarchical_sound_datasets/ESC-50'
    partioned_dataset_dirpath = \
        '/mnt/data/hierarchical_sound_datasets/ESC50_partitioned'
    wav_timelength = 4
    # convert_ESC50_to_hierarchical_dataset(input_dirpath,
    #                                       output_dirpath,
    #                                       wav_timelength)
    partition_dataset(output_dirpath,
                      partioned_dataset_dirpath)

if __name__ == "__main__":
    example_of_usage()
