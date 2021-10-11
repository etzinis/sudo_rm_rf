"""!
@brief Log audio files from a given batch in cometml experiment

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import numpy as np
from scipy.io.wavfile import write as wavwrite


class AudioLogger(object):
    def __init__(self, fs=8000, bs=4, n_sources=2):
        """
        :param dirpath: The path where the audio would be saved.
        :param fs: The sampling rate of the audio in Hz
        :param n_sources: The number of sources
        """
        self.fs = int(fs)
        self.n_sources = int(n_sources)

    def log_batch(self,
                  pr_batch,
                  t_batch,
                  mix_batch,
                  experiment,
                  tag='',
                  step=None):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensor of size:
                         batch_size x num_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensor of size:
                        batch_size x num_sources x length_of_wavs
        :param mix_batch: Batch of the mixtures: Torch Tensor of size:
                          batch_size x 1 x length_of_wavs
        :param experiment: Cometml experiment object
        :param step: The step that this batch belongs
        """
        print('Logging audio online...\n')
        mixture = mix_batch.detach().cpu().numpy()
        true_sources = t_batch.detach().cpu().numpy()
        pred_sources = pr_batch.detach().cpu().numpy()

        # Normalize the audio
        mixture = mixture / np.abs(mixture).max(-1, keepdims=True)
        true_sources = true_sources / np.abs(true_sources).max(-1, keepdims=True)
        pred_sources = pred_sources / np.abs(pred_sources).max(-1, keepdims=True)

        for b_ind in range(mixture.shape[0]):
            experiment.log_audio(mixture[b_ind].squeeze(),
                                 sample_rate=self.fs,
                                 file_name=tag+'batch_{}_mixture'.format(b_ind+1),
                                 metadata=None, overwrite=True,
                                 copy_to_tmp=True, step=step)
            for s_ind in range(self.n_sources):
                experiment.log_audio(
                    true_sources[b_ind][s_ind].squeeze(),
                    sample_rate=self.fs,
                    file_name=tag+'batch_{}_source_{}_true.wav'.format(b_ind+1,
                                                                       s_ind+1),
                    metadata=None, overwrite=True,
                    copy_to_tmp=True, step=step)
                experiment.log_audio(
                    pred_sources[b_ind][s_ind].squeeze(),
                    sample_rate=self.fs,
                    file_name=tag+'batch_{}_source_{}_est.wav'.format(b_ind+1,
                                                                      s_ind+1),
                    metadata=None, overwrite=True,
                    copy_to_tmp=True, step=step)
