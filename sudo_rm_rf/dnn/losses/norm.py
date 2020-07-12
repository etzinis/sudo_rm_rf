"""!
@brief Loss computed under some norm for the estimation of a signal or a mask.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import torch.nn as nn
import itertools


class PermInvariantNorm(nn.Module):
    """!
    Class for estimation under norm computation between estimated signals
    and target signals."""

    def __init__(self,
                 batch_size=None,
                 n_sources=2,
                 weighted_norm=0.0):
        """
        Initialization for the results and torch tensors that might
        be used afterwards

        :param batch_size: The number of the samples in each batch
        :param zero_mean: If you want to perform zero-mean across
        last dimension (time dim) of the signals before SDR computation
        """
        super().__init__()
        self.bs = batch_size
        self.weighted_norm = weighted_norm
        self.permutations = list(itertools.permutations(
            torch.arange(n_sources)))

    def forward(self,
                pr_batch,
                t_batch,
                weights=None,
                eps=1e-7):
        """!

        :param pr_batch: Estimated signals: Torch Tensors of size:
                         batch_size x ...
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x ...
        :param eps: Numerical stability constant

        :returns normed loss for both forward and backward passes.
        """
        mse_l = []
        for perm in self.permutations:
            permuted_pr_batch = (pr_batch[:, perm, :])
            if weights is None:
                se = torch.abs((t_batch ** self.weighted_norm)
                               * (permuted_pr_batch - t_batch))
            else:
                se = torch.abs((weights ** self.weighted_norm)
                               * (permuted_pr_batch - t_batch))
            mse = torch.mean(se.view(se.shape[0], -1), dim=1)
            mse_l.append(mse)

        all_mses = torch.stack(mse_l, dim=1)
        perm_inv_mses = torch.min(all_mses.mean(-2), -1)[0].mean()
        return perm_inv_mses


