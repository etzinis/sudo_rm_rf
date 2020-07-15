"""!
@brief SISNR very efficient computation in Torch. PArts have been used from
asteroid for values cross checking.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import torch.nn as nn
import itertools
from torch.nn.modules.loss import _Loss


def _sdr( y, z, SI=False):
    if SI:
        a = ((z*y).mean(-1) / (y*y).mean(-1)).unsqueeze(-1) * y
        return 10*torch.log10( (a**2).mean(-1) / ((a-z)**2).mean(-1))
    else:
        return 10*torch.log10( (y*y).mean(-1) / ((y-z)**2).mean(-1))

# Negative SDRi loss
def sdri_loss( y, z, of=0):
    # Add a batch dimension if it's missing
    if len( y.shape) < 3:
        y = y.unsqueeze(0)
    if len( z.shape) < 3:
        z = z.unsqueeze(0)

    s = _sdr( y, z, SI=False) - of
    return -s.mean()

# Negative SI-SDRi loss
def sisdr_loss( y, z, of=0):
    # Add a batch dimension if it's missing
    if len( y.shape) < 3:
        y = y.unsqueeze(0)
    if len( z.shape) < 3:
        z = z.unsqueeze(0)

    s = _sdr( y, z, SI=True) - of
    return -s.mean()

# Negative PIT loss
def pit_loss( y, z, of=0, SI=False):
    # Add a batch dimension if it's missing
    if len( y.shape) < 3:
        y = y.unsqueeze(0)
    if len( z.shape) < 3:
        z = z.unsqueeze(0)

    # Get all possible target source permutation SDRs and stack them
    p = list( itertools.permutations( range( y.shape[-2])))
    s = torch.stack( [_sdr( y[:,j,:], z, SI) for j in p], dim=2)

    # Get source-average SDRi
    # s = (s - of.unsqueeze(2)).mean(1)
    s = s.mean(1)

    # Find and return permutation with highest SDRi (negate since we are minimizing)
    i = s.argmax(-1)
    j = torch.arange( s.shape[0], dtype=torch.long, device=i.device)
    return -s[j,i].mean()


class PermInvariantSISDR(nn.Module):
    """!
    Class for SISDR computation between reconstructed signals and
    target wavs by also regulating it with learned target masks."""

    def __init__(self,
                 batch_size=None,
                 zero_mean=False,
                 n_sources=None,
                 backward_loss=True,
                 improvement=False,
                 return_individual_results=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards

        :param batch_size: The number of the samples in each batch
        :param zero_mean: If you want to perform zero-mean across
        last dimension (time dim) of the signals before SDR computation
        """
        super().__init__()
        self.bs = batch_size
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.permutations = list(itertools.permutations(
            torch.arange(n_sources)))
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.improvement = improvement
        self.n_sources = n_sources
        self.return_individual_results = return_individual_results

    def normalize_input(self, pr_batch, t_batch, initial_mixtures=None):
        min_len = min(pr_batch.shape[-1],
                      t_batch.shape[-1])
        if initial_mixtures is not None:
            min_len = min(min_len, initial_mixtures.shape[-1])
            initial_mixtures = initial_mixtures[:, :, :min_len]
        pr_batch = pr_batch[:, :, :min_len]
        t_batch = t_batch[:, :, :min_len]

        if self.perform_zero_mean:
            pr_batch = pr_batch - torch.mean(
                pr_batch, dim=-1, keepdim=True)
            t_batch = t_batch - torch.mean(
                t_batch, dim=-1, keepdim=True)
            if initial_mixtures is not None:
                initial_mixtures = initial_mixtures - torch.mean(
                    initial_mixtures, dim=-1, keepdim=True)
        return pr_batch, t_batch, initial_mixtures

    @staticmethod
    def dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_permuted_sisnrs(self,
                                permuted_pr_batch,
                                t_batch,
                                t_t_diag, eps=10e-8):
        s_t = (self.dot(permuted_pr_batch, t_batch) /
               (t_t_diag + eps) * t_batch)
        e_t = permuted_pr_batch - s_t
        sisnrs = 10 * torch.log10(self.dot(s_t, s_t) /
                                  (self.dot(e_t, e_t) + eps))
        return sisnrs

    def compute_sisnr(self,
                      pr_batch,
                      t_batch,
                      initial_mixtures=None,
                      eps=10e-8):

        t_t_diag = self.dot(t_batch, t_batch)

        sisnr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            sisnr = self.compute_permuted_sisnrs(permuted_pr_batch,
                                                 t_batch,
                                                 t_t_diag, eps=eps)
            sisnr_l.append(sisnr)
        all_sisnrs = torch.cat(sisnr_l, -1)
        best_sisdr, best_perm_ind = torch.max(all_sisnrs.mean(-2), -1)

        if self.improvement:
            initial_mix = initial_mixtures.repeat(1, self.n_sources, 1)
            base_sisdr = self.compute_permuted_sisnrs(initial_mix,
                                                      t_batch,
                                                      t_t_diag, eps=eps)
            best_sisdr -= base_sisdr.mean()

        if not self.return_individual_results:
            best_sisdr = best_sisdr.mean()

        if self.backward_loss:
            return -best_sisdr, best_perm_ind
        return best_sisdr, best_perm_ind

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-9,
                initial_mixtures=None,
                return_best_permutation=False):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
                         batch_size x self.n_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x self.n_sources x length_of_wavs
        :param eps: Numerical stability constant.
        :param initial_mixtures: Initial Mixtures for SISDRi: Torch Tensor
                                 of size: batch_size x 1 x length_of_wavs

        :returns results_buffer Buffer for loading the results directly
                 to gpu and not having to reconstruct the results matrix: Torch
                 Tensor of size: batch_size x 1
        """
        pr_batch, t_batch, initial_mixtures = self.normalize_input(
            pr_batch, t_batch, initial_mixtures=initial_mixtures)

        sisnr_l, best_perm_ind = self.compute_sisnr(
            pr_batch, t_batch, eps=eps,
            initial_mixtures=initial_mixtures)

        if return_best_permutation:
            best_permutations = self.permutations_tensor[best_perm_ind]
            return sisnr_l, best_permutations
        else:
            return sisnr_l

# The following is copied from:
# https://github.com/mpariente/asteroid/blob/master/asteroid/losses
class PITLossWrapper(nn.Module):
    """ Permutation invariant loss wrapper.
    Args:
        loss_func: function with signature (targets, est_targets, **kwargs).
        pit_from (str): Determines how PIT is applied.
            * ``'pw_mtx'`` (pairwise matrix): `loss_func` computes pairwise
              losses and returns a torch.Tensor of shape
              :math:`(batch, n\_src, n\_src)`. Each element
              :math:`[batch, i, j]` corresponds to the loss between
              :math:`targets[:, i]` and :math:`est\_targets[:, j]`
            * ``'pw_pt'`` (pairwise point): `loss_func` computes the loss for
              a batch of single source and single estimates (tensors won't
              have the source axis). Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.get_pw_losses`.
            * ``'perm_avg'``(permutation average): `loss_func` computes the
              average loss for a given permutations of the sources and
              estimates. Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.best_perm_from_perm_avg_loss`.
            In terms of efficiency, ``'perm_avg'`` is the least efficicient.
        perm_reduce (Callable): torch function to reduce permutation losses.
            Defaults to None (equivalent to mean). Signature of the func
            (pwl_set, **kwargs) : (B, n_src!, n_src) --> (B, n_src!).
            `perm_reduce` can receive **kwargs during forward using the
            `reduce_kwargs` argument (dict). If those argument are static,
            consider defining a small function or using `functools.partial`.
            Only used in `'pw_mtx'` and `'pw_pt'` `pit_from` modes.
    For each of these modes, the best permutation and reordering will be
    automatically computed.
    Examples:
        >>> import torch
        >>> from asteroid.losses import pairwise_neg_sisdr
        >>> sources = torch.randn(10, 3, 16000)
        >>> est_sources = torch.randn(10, 3, 16000)
        >>> # Compute PIT loss based on pairwise losses
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        >>> loss_val = loss_func(est_sources, sources)
        >>>
        >>> # Using reduce
        >>> def reduce(perm_loss, src):
        >>>     weighted = perm_loss * src.norm(dim=-1, keepdim=True)
        >>>     return torch.mean(weighted, dim=-1)
        >>>
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx',
        >>>                            perm_reduce=reduce)
        >>> reduce_kwargs = {'src': sources}
        >>> loss_val = loss_func(est_sources, sources,
        >>>                      reduce_kwargs=reduce_kwargs)
    """
    def __init__(self, loss_func, pit_from='pw_mtx', perm_reduce=None):
        super().__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.perm_reduce = perm_reduce
        if self.pit_from not in ['pw_mtx', 'pw_pt', 'perm_avg']:
            raise ValueError('Unsupported loss function type for now. Expected'
                             'one of [`pw_mtx`, `pw_pt`, `perm_avg`]')

    def forward(self, est_targets, targets, return_est=False,
                reduce_kwargs=None, **kwargs):
        """ Find the best permutation and return the loss.
        Args:
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets
            return_est: Boolean. Whether to return the reordered targets
                estimates (To compute metrics or to save example).
            reduce_kwargs (dict or None): kwargs that will be passed to the
                pairwise losses reduce function (`perm_reduce`).
            **kwargs: additional keyword argument that will be passed to the
                loss function.
        Returns:
            - Best permutation loss for each batch sample, average over
                the batch. torch.Tensor(loss_value)
            - The reordered targets estimates if return_est is True.
                torch.Tensor of shape [batch, nsrc, *].
        """
        n_src = targets.shape[1]
        assert n_src < 10, f"Expected source axis along dim 1, found {n_src}"
        if self.pit_from == 'pw_mtx':
            # Loss function already returns pairwise losses
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.pit_from == 'pw_pt':
            # Compute pairwise losses with a for loop.
            pw_losses = self.get_pw_losses(self.loss_func, est_targets,
                                           targets, **kwargs)
        elif self.pit_from == 'perm_avg':
            # Cannot get pairwise losses from this type of loss.
            # Find best permutation directly.
            min_loss, min_loss_idx = self.best_perm_from_perm_avg_loss(
                self.loss_func, est_targets, targets, **kwargs
            )
            # Take the mean over the batch
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
            return mean_loss, reordered
        else:
            return

        assert pw_losses.ndim == 3, ("Something went wrong with the loss "
                                     "function, please read the docs.")
        assert (pw_losses.shape[0] ==
                targets.shape[0]), "PIT loss needs same batch dim as input"

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, min_loss_idx = self.find_best_perm(
            pw_losses, n_src, perm_reduce=self.perm_reduce, **reduce_kwargs
        )
        mean_loss = torch.mean(min_loss)
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
        return mean_loss, reordered

    @staticmethod
    def get_pw_losses(loss_func, est_targets, targets, **kwargs):
        """ Get pair-wise losses between the training targets and its estimate
        for a given loss function.
        Args:
            loss_func: function with signature (targets, est_targets, **kwargs)
                The loss function to get pair-wise losses from.
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets.
            **kwargs: additional keyword argument that will be passed to the
                loss function.
        Returns:
            torch.Tensor or size [batch, nsrc, nsrc], losses computed for
            all permutations of the targets and est_targets.
        This function can be called on a loss function which returns a tensor
        of size [batch]. There are more efficient ways to compute pair-wise
        losses using broadcasting.
        """
        batch_size, n_src, *_ = targets.shape
        pair_wise_losses = targets.new_empty(batch_size, n_src, n_src)
        for est_idx, est_src in enumerate(est_targets.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, est_idx, target_idx] = loss_func(
                    est_src, target_src, **kwargs)
        return pair_wise_losses

    @staticmethod
    def find_best_perm(pair_wise_losses, n_src, perm_reduce=None, **kwargs):
        """Find the best permutation, given the pair-wise losses.
        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape [batch, n_src, n_src]. Pairwise losses.
            n_src (int): Number of sources.
            perm_reduce (Callable): torch function to reduce permutation losses.
                Defaults to None (equivalent to mean). Signature of the func
                (pwl_set, **kwargs) : (B, n_src!, n_src) --> (B, n_src!)
            **kwargs: additional keyword argument that will be passed to the
                permutation reduce function.
        Returns:
            tuple:
                :class:`torch.Tensor`: The loss corresponding to the best
                permutation of size (batch,).
                :class:`torch.LongTensor`: The indexes of the best permutations.
        MIT Copyright (c) 2018 Kaituo XU.
        See `Original code
        <https://github.com/kaituoxu/Conv-TasNet/blob/master>`__ and `License
        <https://github.com/kaituoxu/Conv-TasNet/blob/master/LICENSE>`__.
        """
        # After transposition, dim 1 corresp. to sources and dim 2 to estimates
        pwl = pair_wise_losses.transpose(-1, -2)
        perms = pwl.new_tensor(list(itertools.permutations(range(n_src))),
                               dtype=torch.long)
        # Column permutation indices
        idx = torch.unsqueeze(perms, 2)
        # Loss mean of each permutation
        if perm_reduce is None:
            # one-hot, [n_src!, n_src, n_src]
            perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2,
                                                                           idx,
                                                                           1)
            loss_set = torch.einsum('bij,pij->bp', [pwl, perms_one_hot])
            loss_set /= n_src
        else:
            batch = pwl.shape[0]
            n_perm = idx.shape[0]
            # [batch, n_src!, n_src] : Pairwise losses for each permutation.
            pwl_set = pwl[:, torch.arange(n_src), idx.squeeze(-1)]
            # Apply reduce [batch, n_src!, n_src] --> [batch, n_src!]
            loss_set = perm_reduce(pwl_set, **kwargs)
        # Indexes and values of min losses for each batch element
        min_loss_idx = torch.argmin(loss_set, dim=1)
        min_loss, _ = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_idx


class PairwiseNegSDR(_Loss):
    """ Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.
        Args:
            sdr_type (str): choose between "snr" for plain SNR, "sisdr" for
                SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target
                and estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.
        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape
                [batch, n_src, time]. Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape
                [batch, n_src, time]. Batch of training targets.
        Returns:
            :class:`torch.Tensor`: with shape [batch, n_src, n_src].
            Pairwise losses.
        Examples:
            >>> import torch
            >>> from asteroid.losses import PITLossWrapper
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
            >>>                            pit_from='pairwise')
            >>> loss = loss_func(est_targets, targets)
        References:
            [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
            International Conference on Acoustics, Speech and Signal
            Processing (ICASSP) 2019.
        """
    def __init__(self, sdr_type, zero_mean=True, take_log=True):
        super(PairwiseNegSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_targets, targets):
        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)

        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3,
                                      keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target**2, dim=3, keepdim=True) + 1e-8
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
                torch.sum(e_noise ** 2, dim=3) + 1e-8)
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + 1e-8)
        return - pair_wise_sdr