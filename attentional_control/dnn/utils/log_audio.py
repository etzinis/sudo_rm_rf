"""!
@brief Log audio files from a given batch

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import numpy as np
from scipy.io.wavfile import write as wavwrite


class AudioLogger(object):
    def __init__(self, dirpath, fs, bs, n_sources):
        """
        :param dirpath: The path where the audio would be saved.
        :param fs: The sampling rate of the audio in Hz
        :param bs: The number of samples in batch
        :param n_sources: The number of sources
        """
        if dirpath is not None:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
        else:
            raise NotADirectoryError("You try to specify an AudioLogger "
                                     "without specifying a valid path!")
        self.dirpath = os.path.abspath(dirpath)
        self.fs = int(fs)
        self.bs = int(bs)
        self.n_sources = int(n_sources)

    def log_batch(self,
                  pr_batch,
                  t_batch,
                  mix_batch,
                  mixture_rec=None):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensor of size:
                         batch_size x num_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensor of size:
                        batch_size x num_sources x length_of_wavs
        :param mix_batch: Batch of the mixtures: Torch Tensor of size:
                          batch_size x 1 x length_of_wavs
        :param mixture_rec: Batch of the reconstructed mixtures: Torch Tensor of
                            size: batch_size x 1 x length_of_wavs
        """


        mixture = mix_batch.detach().cpu().numpy()
        true_sources = t_batch.detach().cpu().numpy()
        pred_sources = pr_batch.detach().cpu().numpy()

        for b_ind in range(self.bs):
            mix_name = "bind_{}_mix.wav".format(b_ind)
            mix_wav = (mixture[b_ind][0] /
                       (np.max(np.abs(mixture[b_ind][0])) + 10e-8))
            wavwrite(os.path.join(self.dirpath, mix_name),
                     self.fs,
                     mix_wav)
            for s_ind in range(self.n_sources):
                true_s_name = "bind_{}_true_s{}.wav".format(b_ind, s_ind)
                rec_s_name = "bind_{}_rec_s{}.wav".format(b_ind, s_ind)
                rec_wav = (pred_sources[b_ind][s_ind] /
                           (np.max(np.abs(pred_sources[b_ind][s_ind])) + 10e-8))
                true_wav = (true_sources[b_ind][s_ind] /
                           (np.max(np.abs(true_sources[b_ind][s_ind])) + 10e-8))
                wavwrite(os.path.join(self.dirpath, true_s_name),
                         self.fs,
                         true_wav)
                wavwrite(os.path.join(self.dirpath, rec_s_name),
                         self.fs,
                         rec_wav)

        if mixture_rec is not None:
            mixture_rec_np = mixture_rec.detach().cpu().numpy()
            for b_ind in range(self.bs):
                rec_mix_name = "bind_{}_rec_mix.wav".format(b_ind)
                rec_mix_wav = (mixture_rec_np[b_ind][0] /
                               (np.max(np.abs(mixture_rec_np[b_ind][0])) +
                                10e-8))
                wavwrite(os.path.join(self.dirpath, rec_mix_name),
                         self.fs,
                         rec_mix_wav)
