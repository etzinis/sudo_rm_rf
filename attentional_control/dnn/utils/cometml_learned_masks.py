"""!
@brief Library for experiment loss functionality

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')


def log_one_heatmap(experiment,
                    cmap,
                    title,
                    np_array):
    if np_array.shape[1] > 3 * np_array.shape[0]:
        many = int(np_array.shape[1] / (1. * np_array.shape[0]))
        ar = np.repeat(np_array, [many for _ in range(np_array.shape[0])], axis=0)
    elif np_array.shape[0] > 3 * np_array.shape[1]:
        many = int(np_array.shape[0] / (1. * np_array.shape[1]))
        ar = np.repeat(np_array, [many for _ in range(np_array.shape[1])],
                       axis=1)
    else:
        ar = np_array
    plt.imshow(ar, cmap=cmap,
               interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    experiment.log_figure(figure=plt, figure_name=title, overwrite=False)
    plt.close()


def log_heatmaps(experiment,
                 np_images,
                 titles,
                 cmaps):
    assert len(cmaps) == len(titles)
    assert len(titles) == len(np_images)

    for np_image, title, cmap in zip(np_images, titles, cmaps):
        log_one_heatmap(experiment,
                        cmap,
                        title,
                        np_image)


def create_and_log_afe_internal(experiment,
                                enc_masks,
                                mix_encoded,
                                encoder_basis,
                                decoder_basis):

    np_images = [enc_masks[0], enc_masks[1], mix_encoded,
                 encoder_basis, decoder_basis]
    cmaps = ['Blues', 'Reds', 'jet', 'Greens', 'Oranges']
    titles = ['Mask 1', 'Mask 2', 'Encoded Mix', 'Encoder Basis',
              'Decoder Basis']

    log_heatmaps(experiment, np_images, titles, cmaps)

def create_and_log_tasnet_masks(experiment,
                                recon_masks,
                                target_masks,
                                mix_encoded,
                                encoder_basis,
                                decoder_basis):
        # Check also the difference between the reconstructed masks and their
        # target pairs.

        diff00 = np.abs(target_masks[0] - recon_masks[0])
        diff01 = np.abs(target_masks[0] - recon_masks[1])
        diff10 = np.abs(target_masks[1] - recon_masks[0])
        diff11 = np.abs(target_masks[1] - recon_masks[1])

        if diff00.sum() < diff01.sum():
            diff_s1 = diff00
            diff_s2 = diff11
        else:
            diff_s1 = diff01
            diff_s2 = diff10

        np_images = [recon_masks[0], recon_masks[1],
                     target_masks[0], target_masks[1],
                     diff_s1, diff_s2,
                     mix_encoded, encoder_basis, decoder_basis]
        cmaps = ['Blues', 'Reds', 'Purples', 'Oranges',
                 'binary', 'binary_r',
                 'jet', 'Greens', 'viridis']
        titles = ['Rec Mask 1', 'Rec Mask 2', 'Target Mask 1', 'Target Mask 2',
                  'Diff Mask 1', 'Diff Mask 2',
                  'Encoded Mix', 'Encoder Basis', 'Decoder Basis']

        log_heatmaps(experiment, np_images, titles, cmaps)
