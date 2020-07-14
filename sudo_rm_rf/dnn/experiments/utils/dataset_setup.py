"""!
@brief Infer Dataset Specific parameters and return generators

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

from __config__ import WHAM_ROOT_PATH
import sudo_rm_rf.dnn.dataset_loader.wham as wham_loader


def create_loader_for_simple_dataset(dataset_name=None,
                                     separation_task=None,
                                     data_split=None,
                                     sample_rate=None,
                                     min_or_max=None,
                                     zero_pad=None,
                                     timelegth=None,
                                     normalize_audio=None,
                                     n_samples=None):
    if dataset_name == 'WHAM':
        loader = wham_loader
        root_path = WHAM_ROOT_PATH
        translator = {'train': 'tr', 'test': 'tt', 'val': 'cv'}
        translated_split = translator[data_split]
    else:
        raise ValueError('Dataset: {} is not yet supported!'.format(
            dataset_name))

    data_loader = loader.Dataset(
        root_dirpath=root_path, task=separation_task,
        split=translated_split, sample_rate=sample_rate, timelength=timelegth,
        zero_pad=zero_pad, min_or_max=min_or_max,
        augment='tr' in data_split,
        normalize_audio=normalize_audio, n_samples=n_samples)
    return data_loader

def setup(hparams):
    # Create all generators
    generators = {}
    for data_split in ['train', 'val', 'test', 'train_val']:
        if hparams[data_split] is None:
            generators[data_split] = None
            continue

        if len(hparams[data_split]) > 1:
            raise ValueError('Current implementation does not support '
                             'training using multiple datasets.')

        loader = create_loader_for_simple_dataset(
                    dataset_name=hparams[data_split][0],
                    separation_task=hparams['separation_task'],
                    data_split=data_split.split('_')[0],
                    sample_rate=hparams['fs'],
                    min_or_max=hparams['min_or_max'],
                    zero_pad=hparams['zero_pad_audio'],
                    timelegth=hparams['audio_timelength'],
                    normalize_audio=hparams['normalize_audio'],
                    n_samples=hparams['n_'+data_split])
        generators[data_split] = loader.get_generator(
            batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])

    return generators
