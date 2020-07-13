"""!
@brief Infer Dataset Specific parameters

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""
import argparse
import os
import sys

sys.path.append('../../../../')
import sudo_rm_rf.dnn.dataset_loader.torch_dataloader as dataloader
import sudo_rm_rf.dnn.dataset_loader.augmented_mix_dataloader as \
    augmented_dataloader
from __config__ import *


def is_augmented_dataset(datasets):
    any_augmented = any(['AUGMENTED' in d for d in datasets])
    all_augmented = all(['AUGMENTED' in d for d in datasets])

    if any_augmented and not all_augmented:
        raise ValueError('If one dataset is augmented one then all the other '
                         'on the same partition must also be augmented.')
    elif not any_augmented and not all_augmented:
        return False
    else:
        return True


def infer_augmented_dataset_name(datasets, priors):
    datasets_names = [d.split('_')[-1] for d in datasets]
    return '_'.join([d + '_' + str(priors[i])
                     for (i, d) in enumerate(datasets_names)])


def get_hierarchical_dataset_rootdir(dataset_name):
    if 'WSJ' in dataset_name:
        return WSJ_MIX_HIERARCHICAL_P
    elif 'ESC50' in dataset_name:
        return ESC50_HIERARCHICAL_P
    else:
        raise NotImplementedError('Hierarchical dataset: {} is not yet '
                                  'available'.format(dataset_name))


def update_hparams(hparams):
    # Check whether there is an augmented dataset
    if is_augmented_dataset(hparams['train_dataset']):
        train_dataset_name = infer_augmented_dataset_name(
            hparams['train_dataset'], hparams['datasets_priors'])
        hparams['augmented_dataset_name'] = train_dataset_name
        hparams['in_samples'] = 32000
        hparams['n_sources'] = 2
        hparams['fs'] = 8000.
        hparams['train_dataset_path'] = [
            os.path.join(get_hierarchical_dataset_rootdir(d_name), 'train')
            for d_name in hparams['train_dataset']]
        hparams['return_items'] = ['mixture_wav_norm',
                                   'clean_sources_wavs_norm']

        if is_augmented_dataset(hparams['val_dataset']):
            hparams['val_dataset_path'] = [
                os.path.join(get_hierarchical_dataset_rootdir(d_name), 'test')
                for d_name in hparams['val_dataset']]
        else:
            hparams['val_dataset'] = hparams['val_dataset'][0]

        if hparams['train_val_dataset'] is None:
            return

        if is_augmented_dataset(hparams['train_val_dataset']):
            hparams['train_val_dataset_path'] = [
                os.path.join(get_hierarchical_dataset_rootdir(d_name), 'train')
                for d_name in hparams['train_val_dataset']]
        else:
            hparams['train_val_dataset'] = hparams['train_val_dataset'][0]
        return

    if len(hparams['train_dataset']) == 1:
        hparams['train_dataset'] = hparams['train_dataset'][0]
    if len(hparams['val_dataset']) == 1:
        hparams['val_dataset'] = hparams['val_dataset'][0]
    if len(hparams['train_val_dataset']) == 1:
        hparams['train_val_dataset'] = hparams['train_val_dataset'][0]


def get_data_loaders(hparams):

    # Augmented dataloader return items
    if hparams['class_loss_weight'] != 0:
        hparams['return_items'] = ['wav', 'class_id']
    else:
        hparams['return_items'] = ['wav']

    if isinstance(hparams['train_dataset_path'], str):
        train_gen = dataloader.get_data_generators(
            [hparams['train_dataset_path']], bs=hparams['bs'],
            n_jobs=hparams['n_jobs'], get_top=[hparams['tr_get_top']],
            return_items=hparams['return_items'])[0]
    else:
        these_args = argparse.Namespace(
            input_dataset_p=hparams['train_dataset_path'],
            datasets_priors=hparams['datasets_priors'],
            batch_size=hparams['bs'],
            n_jobs=hparams['n_jobs'],
            n_samples=hparams['tr_get_top'],
            return_items=hparams['return_items'],
            fs=hparams['fs'],
            selected_timelength=hparams['selected_timelength'],
            n_sources=hparams['n_sources'],
            max_abs_snr=hparams['max_abs_snr'],
            fixed_seed=hparams['fixed_seed']
        )

        data_loader = augmented_dataloader.AugmentedOnlineMixingDataset(
            **vars(these_args))
        train_gen = augmented_dataloader.get_data_gen_from_loader(data_loader)

    if isinstance(hparams['val_dataset_path'], str):
        val_gen = dataloader.get_data_generators(
            [hparams['val_dataset_path']], bs=hparams['bs'],
            n_jobs=hparams['n_jobs'], get_top=[hparams['tr_get_top']],
            return_items=hparams['return_items'])[0]
    else:
        these_args = argparse.Namespace(
            input_dataset_p=hparams['val_dataset_path'],
            datasets_priors=hparams['datasets_priors'],
            batch_size=hparams['bs'],
            n_jobs=hparams['n_jobs'],
            n_samples=hparams['val_get_top'],
            return_items=hparams['return_items'],
            fs=hparams['fs'],
            selected_timelength=hparams['selected_timelength'],
            n_sources=hparams['n_sources'],
            max_abs_snr=hparams['max_abs_snr'],
            fixed_seed=7
        )

        data_loader = augmented_dataloader.AugmentedOnlineMixingDataset(
            **vars(these_args))
        val_gen = augmented_dataloader.get_data_gen_from_loader(data_loader)

    if hparams['train_val_dataset'] is None:
        train_val_gen = None
    elif isinstance(hparams['train_val_dataset_path'], str):
        train_val_gen = dataloader.get_data_generators(
            [hparams['train_val_dataset_path']], bs=hparams['bs'],
            n_jobs=hparams['n_jobs'], get_top=[hparams['val_get_top']],
            return_items=hparams['return_items'])[0]
    else:
        these_args = argparse.Namespace(
            input_dataset_p=hparams['train_val_dataset_path'],
            datasets_priors=hparams['datasets_priors'],
            batch_size=hparams['bs'],
            n_jobs=hparams['n_jobs'],
            n_samples=hparams['val_get_top'],
            return_items=hparams['return_items'],
            fs=hparams['fs'],
            selected_timelength=hparams['selected_timelength'],
            n_sources=hparams['n_sources'],
            max_abs_snr=hparams['max_abs_snr'],
            fixed_seed=8
        )

        data_loader = augmented_dataloader.AugmentedOnlineMixingDataset(
            **vars(these_args))
        train_val_gen = augmented_dataloader.get_data_gen_from_loader(data_loader)

    return train_gen, val_gen, train_val_gen