"""!
@brief Testing the validity of the norm loss computation

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import numpy as np
import torch
from time import time
import sys
from argparse import Namespace
sys.path.append("../../../")
import dnn.losses.norm as norm_torch
import itertools


def naive_pit_norm(perm_pr_batch, t_batch, n_sources,
                   eps=10e-8, weighted=False):
    if weighted:
        return np.mean(np.abs(t_batch ** 2 * (perm_pr_batch - t_batch)))
    else:
        return np.mean(np.abs(perm_pr_batch - t_batch))


def naive_norm_loss(pr_batch, t_batch, n_sources, eps=10e-8, weighted=False):
    total = 0.
    permutations = list(itertools.permutations(np.arange(n_sources)))
    for b_ind in np.arange(pr_batch.shape[0]):
        min_norm = None
        for p in permutations:
            perm_pr_batch = pr_batch[b_ind, p, :]
            norm_acc = naive_pit_norm(perm_pr_batch,
                                      t_batch[b_ind],
                                      n_sources,
                                      weighted=weighted)
            if min_norm is None:
                min_norm = norm_acc
            elif norm_acc < min_norm:
                min_norm = norm_acc
        total += min_norm
    return total / pr_batch.shape[0]


def random_batch_creator(n_batches=2,
                         bs=16,
                         n_sources=3,
                         n_freqs=40,
                         n_frames=50):
    return np.asarray(np.random.rand(n_batches, bs, n_sources, n_freqs,
                                     n_frames),
                      dtype=np.float32)


def test_sisnr_implementations(n_batches=1,
                               bs=6,
                               n_sources=5,
                               n_freqs=50,
                               n_frames=160,
                               weighted=True):
    cpu_timer, gpu_timer = 0., 0.
    gpu_results = torch.zeros((n_batches, bs - 1, n_sources)).float()
    cpu_results = np.zeros((n_batches, bs - 1, n_sources))
    cpu_batches = random_batch_creator(n_batches=n_batches,
                                       bs=bs,
                                       n_sources=n_sources,
                                       n_freqs=n_freqs,
                                       n_frames=n_frames)
    gpu_batches = torch.from_numpy(cpu_batches)
    torch.set_printoptions(precision=8)

    for b_ind in np.arange(cpu_batches.shape[0]):
        before = time()
        cpu_results[b_ind, :, :] = \
            naive_norm_loss(cpu_batches[b_ind, :-1, :, :],
                            cpu_batches[b_ind, 1:, :, :],
                            n_sources,
                            weighted=weighted)
        now = time()
        cpu_timer += now - before

    gpu_sisnr = norm_torch.PermInvariantNorm(batch_size=bs,
                                             n_sources=n_sources,
                                             weighted_norm=2.)

    for b_ind in np.arange(gpu_batches.shape[0]):
        before = time()
        gpu_results[b_ind, :, :] = \
            gpu_sisnr(gpu_batches[b_ind, :-1, :, :],
                      gpu_batches[b_ind, 1:, :, :],
                      weights=gpu_batches[b_ind, 1:, :, :])
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
