"""!
@brief Testing the validity of the sisnr loss computation

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import numpy as np
import torch
from time import time
import sys
from argparse import Namespace
sys.path.append("../../../")
import dnn.losses.sisdr as sisdr_l
import itertools


def naive_perm_sisnr(perm_pr_batch, t_batch, n_sources, eps=10e-8):
    sisnr_acc = 0.
    for s_ind in np.arange(n_sources):
        s_hat = perm_pr_batch[s_ind]
        s = t_batch[s_ind]
        s_hat = s_hat - np.mean(s_hat)
        s = s - np.mean(s)
        s_t = np.dot(s_hat, s) / (np.dot(s, s) + eps) * s
        e_t = s_hat - s_t
        sisnr = 10 * np.log10(np.dot(s_t, s_t) / (np.dot(e_t, e_t) + eps))
        sisnr_acc += sisnr / n_sources
    return sisnr_acc

def naive_sisnr(pr_batch, t_batch, n_sources, eps=10e-8, improvement=False):
    total = 0.
    permutations = list(itertools.permutations(np.arange(n_sources)))
    for b_ind in np.arange(pr_batch.shape[0]):
        max_sisnr = None
        for p in permutations:
            perm_pr_batch = pr_batch[b_ind, p, :]
            sisnr_acc = naive_perm_sisnr(perm_pr_batch,
                                         t_batch[b_ind],
                                         n_sources)
            if max_sisnr is None:
                max_sisnr = sisnr_acc
            elif max_sisnr < sisnr_acc:
                max_sisnr = sisnr_acc
        total += max_sisnr / pr_batch.shape[0]
        if improvement:
            initial_mix = np.sum(t_batch[b_ind], axis=0, keepdims=0)
            initial_mix = np.repeat(initial_mix, n_sources, axis=0)
            base_sisdr = naive_perm_sisnr(initial_mix,
                                          t_batch[b_ind],
                                          n_sources)
            print("CPU")
            print(base_sisdr)
            total -= base_sisdr.mean()

    return total


def random_batch_creator(n_batches=2,
                         bs=16,
                         n_sources=2,
                         length=16000):
    return np.asarray(np.random.rand(n_batches, bs, n_sources, length),
                      dtype=np.float32)


def test_sisnr_implementations(n_batches=2,
                               bs=5,
                               n_sources=1,
                               length=16000,
                               improvement=False):
    cpu_timer, gpu_timer = 0., 0.
    gpu_results = torch.zeros((n_batches, bs - 1, n_sources)).float()
    cpu_results = np.zeros((n_batches, bs - 1, n_sources))
    cpu_batches = random_batch_creator(n_batches=n_batches,
                                       bs=bs,
                                       n_sources=n_sources,
                                       length=length)
    gpu_batches = torch.from_numpy(cpu_batches)
    torch.set_printoptions(precision=8)

    for b_ind in np.arange(cpu_batches.shape[0]):
        before = time()
        cpu_results[b_ind, :, :] = \
            naive_sisnr(cpu_batches[b_ind, :-1, :, :],
                        cpu_batches[b_ind, 1:, :, :],
                        n_sources,
                        improvement=improvement)
        now = time()
        cpu_timer += now - before

    gpu_sisnr = sisdr_l.PermInvariantSISDR(batch_size=bs,
                                           n_sources=n_sources,
                                           zero_mean=True,
                                           backward_loss=False,
                                           improvement=improvement)

    for b_ind in np.arange(gpu_batches.shape[0]):
        before = time()
        pr_batch = gpu_batches[b_ind, :-1, :, :]
        t_batch = gpu_batches[b_ind, 1:, :, :]
        gpu_results[b_ind, :, :] = \
            gpu_sisnr(pr_batch, t_batch,
                      initial_mixtures=torch.sum(
                          t_batch, 1, keepdims=True))
        now = time()
        gpu_timer += now - before

    print("CPU")
    print(cpu_results)
    print("GPU")
    print(gpu_results.data.cpu().numpy())
    print("DIFF")
    print(np.abs(cpu_results - gpu_results.data.cpu().numpy()))


if __name__ == "__main__":
    test_sisnr_implementations()
