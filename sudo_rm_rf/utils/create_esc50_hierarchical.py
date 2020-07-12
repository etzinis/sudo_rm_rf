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
import csv
from tqdm import tqdm
import shutil

import sudo_rm_rf.utils.progress_display as progress_display

from __config__ import ESC50_DOWNLOADED_P, ESC50_HIERARCHICAL_P


def normalize_wav(wav, eps=10e-7, std=None):
    mean = wav.mean()
    if std is None:
        std = wav.std()
    return (wav - mean) / (std + eps)


def write_data_wrapper_func(audio_files_dir,
                            sound_classes,
                            min_wav_timelength,
                            output_dirpath,
                            metadata_dict):
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

        # Append metadata to the written data
        for meta_label, meta_label_val in metadata_dict[uid].items():
            if meta_label in data:
                raise IndexError('Trying to override essential '
                                 'information about files by '
                                 'assigning metalabel: {} in data '
                                 'dictionary: {}'.format(meta_label,
                                                         data))
            data[meta_label] = meta_label_val

        if not os.path.exists(output_uid_folder):
            os.makedirs(output_uid_folder)

        for k, v in data.items():
            file_path = os.path.join(output_uid_folder, k)
            joblib.dump(v, file_path, compress=0)

    return lambda uid: process_uid(uid)


def getMetaDataDict(meta_csv_filepath):
    try:
        meta_data_dict = {}
        with open(meta_csv_filepath) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                meta_data_dict[row['filename']] = {
                    'fold': torch.tensor(int(row['fold']),
                                         dtype=torch.int32),
                    'class_id': torch.tensor(int(row['target']),
                                             dtype=torch.int32),
                    'human_readable_class': row['category'],
                    'source_file': row['src_file']
                }
            return meta_data_dict
    except Exception as e:
        print('Failed at loading metadata for ESC-50 dataset')
        raise e


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
    if not files:
        raise IOError('Selected folder: {} does not contain any wav '
                      'file.'.format(audio_files_dir))

    meta_csv_filepath = os.path.join(input_dirpath, 'meta/esc50.csv')
    metadata_dict = getMetaDataDict(meta_csv_filepath)

    unique_ids = set([os.path.basename(f) for f in files])

    for x in unique_ids:
        if x not in metadata_dict:
            raise IndexError('Metadata for file: {} not parsed '
                             'correctly!'.format(x))

    sound_classes = set([u.split('-')[-1].split('.wav')[0]
                         for u in unique_ids])

    write_data_func = write_data_wrapper_func(audio_files_dir,
                                              sound_classes,
                                              wav_timelength,
                                              output_dirpath,
                                              metadata_dict)

    progress_display.progress_bar_wrapper(
        write_data_func,
        list(unique_ids),
        message='Processing {} files...'.format(len(unique_ids)))

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

    print("Partitioned Dataset ready at: {}".format(partitions_dirpath))


def example_of_usage():
    input_dirpath = ESC50_DOWNLOADED_P
    temp_dirpath = '/tmp/ESC-50'
    partioned_dataset_dirpath = ESC50_HIERARCHICAL_P
    wav_timelength = 4
    convert_ESC50_to_hierarchical_dataset(input_dirpath,
                                          temp_dirpath,
                                          wav_timelength)
    partition_dataset(temp_dirpath,
                      partioned_dataset_dirpath)

if __name__ == "__main__":
    example_of_usage()
