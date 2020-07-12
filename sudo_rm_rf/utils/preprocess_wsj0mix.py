"""!
@brief Data Preprocessor for wsj0-mix dataset for more efficient
loading and also in order to be able to use the universal pytorch
loader.
@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(root_dir)

from glob2 import glob
import joblib
from scipy.io import wavfile
import torch
import sudo_rm_rf.utils.progress_display as progress_display
from __config__ import *



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
        output_name = "wsj0_{}mix_{}k_{}s_{}_preprocessed".format(
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


def normalize_tensor_wav(wav, eps=10e-7, std=None):
    mean = wav.mean(-1, keepdim=True)
    if std is None:
        std = wav.std(-1, keepdim=True)
    return (wav - mean) / (std + eps)


def write_data_wrapper_func(input_dirpath,
                            clean_folders,
                            max_wav_samples,
                            output_dirpath,
                            min_or_max=None):

    def process_uid(uid):
        mix_path = os.path.join(input_dirpath, 'mix', uid)
        mix_wav = wavfile.read(mix_path)[1] / 29491.

        mix_wav = torch.tensor(mix_wav, dtype=torch.float32)

        sources_paths = [os.path.join(input_dirpath, fo, uid)
                         for fo in clean_folders]
        sources_w_list = [wavfile.read(p)[1] / 29491.
                          for p in sources_paths]
        sources_w_list = [torch.tensor(normalize_wav(np_vec),
                                       dtype=torch.float32).unsqueeze(0)
                          for np_vec in sources_w_list]
        sources_wavs = torch.cat(sources_w_list, dim=0)

        min_len = min(mix_wav.shape[-1], sources_wavs.shape[-1])
        if min_len < max_wav_samples:
            padded_mix_wav = torch.zeros(max_wav_samples)
            padded_sources_wavs = torch.zeros(sources_wavs.shape[0],
                                              max_wav_samples)
            padded_mix_wav[:mix_wav.shape[-1]] = mix_wav
            padded_sources_wavs[
                :, :min(max_wav_samples, sources_wavs.shape[-1])] = \
                sources_wavs[:, :min(max_wav_samples, sources_wavs.shape[-1])]
            mix_wav = padded_mix_wav
            sources_wavs = padded_sources_wavs

        mix_wav = mix_wav[:max_wav_samples]
        sources_wavs = sources_wavs[:, :max_wav_samples]
        mix_std = mix_wav.detach().cpu().numpy().std()
        mix_wav_norm = torch.tensor(normalize_wav(mix_wav),
                                    dtype=torch.float32)
        sources_wavs_norm = normalize_tensor_wav(sources_wavs,
                                                 std=mix_std)

        output_uid_folder = os.path.join(output_dirpath, uid)
        data = {
            'mixture_wav': mix_wav,
            'clean_sources_wavs': sources_wavs,
            'mixture_wav_norm': mix_wav_norm,
            'clean_sources_wavs_norm': sources_wavs_norm
        }

        if not os.path.exists(output_uid_folder):
            os.makedirs(output_uid_folder)

        for k, v in data.items():
            file_path = os.path.join(output_uid_folder, k)
            joblib.dump(v, file_path, compress=0)

    return lambda uid: process_uid(uid)


def convert_subset(input_dirpath,
                   output_dirpath,
                   fs,
                   n_speakers,
                   wav_timelength,
                   subset=None,
                   min_or_max=None):
    """! Convert a subset of files in the appropriate format
    Args:
        input_dirpath: The path of a wsj0mix dataset e.g.
                       wsj0-mix/2speakers/wav8k/min (for mixes with
                       minimum length)
        output_dirpath: The path for storing the new dataset
                        (the directories would be created recursively)
        n_speakers: number of speakers in mixtures
        fs: sampling rate in Hz
        wav_timelength: The timelength in seconds of all the mixtures and
                        clean sources
    Generates: The structure in output_dirpath where each unique id would
    have a corresponding folder with all the files in tensors or .txt for
    configs
            e.g. uid: 423a0105_1.2681_446o030n_-1.2681.wav -->
            output_dirpath/423a0105_1.2681_446o030n_-1.2681
                          mixture_wav (tensor)
                          clean_sources_wavs (tensor)
    """
    mixtures_dir = os.path.join(input_dirpath, 'mix')
    files = glob(mixtures_dir + '/*.wav')
    unique_ids = set([os.path.basename(f) for f in files])
    max_wav_samples = int(wav_timelength * fs)

    clean_folders = ['s'+str(n+1) for n in range(n_speakers)]

    write_data_func = write_data_wrapper_func(input_dirpath,
                                              clean_folders,
                                              max_wav_samples,
                                              output_dirpath,
                                              min_or_max=min_or_max)

    progress_display.progress_bar_wrapper(
        write_data_func,
        list(unique_ids),
        message='{}: Processing {} files...'.format(
            subset, len(unique_ids)))


def convert_wsj0mix_to_universal_dataset(input_dirpath,
                                         output_dirpath,
                                         wav_timelength):
    """! This function converts the wsj0mix dataset in a universal
    dataset where each sample has each own folder and all values are
    stored as Tensors for efficiency and in order to be loaded
    irrespective of the structure of the dataset.
    Args:
        input_dirpath: The path of a wsj0mix dataset e.g.
                       wsj0-mix/2speakers/wav8k/min (for mixes with
                       minimum length)
        output_dirpath: The path for storing the new dataset
                        (the directories would be created recursively)
        wav_timelength: The timelength in seconds of all the mixtures
                        and clean sources
    Intermediate:
        output_name: Default name would be infered as follows:
                     E.g. for wsj0-mix/2speakers/wav8k/min and for 4s
                     timelength:
                     it would be wsj0_2mix_8k_4s_min_preprocessed
    """
    output_name, fs, n_speakers, min_or_max = infer_output_name(
        input_dirpath, wav_timelength)

    root_out_dir = os.path.join(output_dirpath, output_name)

    subsets = os.listdir(input_dirpath)
    print("All the subsets are: {}".format(subsets))
    for subset in subsets:
        subset_input_dirpath = os.path.join(input_dirpath, subset)
        subset_output_dirpath = os.path.join(root_out_dir, subset)
        convert_subset(subset_input_dirpath,
                       subset_output_dirpath,
                       fs,
                       n_speakers,
                       wav_timelength,
                       subset=subset,
                       min_or_max=min_or_max)

    print("Dataset ready at: {}".format(root_out_dir))

def run_all():
    input_dirpath = os.path.join(WSJ0_MIX_2_8K_PATH, 'max')
    output_dirpath = WSJ0_MIX_2_8K_PREPROCESSED_BASE_P
    wav_timelength = 4
    convert_wsj0mix_to_universal_dataset(input_dirpath,
                                         output_dirpath,
                                         wav_timelength)


if __name__ == "__main__":
    run_all()
