"""!
@brief Mixture consistency according to the paper.
Scott Wisdom, John R Hershey, Kevin Wilson, Jeremy Thorpe, Michael
Chinen, Brian Patton, and Rif A Saurous. "Differentiable consistency
constraints for improved deep speech enhancement", ICASSP 2019.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch


def apply(pr_batch, input_mixture, mix_weights_type='uniform'):
    """Apply mixture consistency

    :param pr_batch: Torch Tensors of size:
                     batch_size x self.n_sources x length_of_wavs
    :param input_mixture: Torch Tensors of size:
                     batch_size x 1 x length_of_wavs
    :param mix_weights_type: type of wights applied
    """
    num_sources = pr_batch.shape[1]
    pr_mixture = torch.sum(pr_batch, 1, keepdim=True)

    if mix_weights_type == 'magsq':
        mix_weights = torch.mean(pr_batch ** 2, -1, keepdim=True)
        mix_weights /= (torch.sum(mix_weights, 1, keepdim=True) + 1e-9)
    elif mix_weights_type == 'uniform':
        mix_weights = (1.0 / num_sources)
    else:
        raise ValueError('Invalid mixture consistency weight type: {}'
                         ''.format(mix_weights_type))

    source_correction = mix_weights * (input_mixture - pr_mixture)
    return pr_batch + source_correction
