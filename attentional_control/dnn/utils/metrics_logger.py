"""!
@brief Library for saving metrics per sample and per epoch

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import numpy as np


def log_metrics(metrics_dict, dirpath, tr_step, val_step,
                cometml_experiment=None):
    """Logs the accumulative individual results from a dictionary of metrics

    Args:
        metrics_dict: Python Dict with the following structure:
                     res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        dirpath:  An absolute path for saving the metrics into
        tr_step:     The step/epoch index for training
        val_step:     The step/epoch index for validation
    """

    for metric_name, metric_data in metrics_dict.items():
        this_metric_folder = os.path.join(dirpath, metric_name)
        if not os.path.exists(this_metric_folder):
            print("Creating non-existing metric log directory... {}"
                  "".format(this_metric_folder))
            os.makedirs(this_metric_folder)

        values = metric_data['acc']
        values = np.array(values)
        if metric_name.startswith('tr_'):
            this_step = tr_step
        elif metric_name.startswith('val_'):
            this_step = val_step
        else:
            NotImplementedError('I am not sure where to put this '
                                'metric: {}'.format(metric_name))
        filename = 'epoch_{}.npy'.format(this_step)
        savepath = os.path.join(this_metric_folder, filename)
        np.save(savepath, values)

        if cometml_experiment is not None:
            cometml_experiment.log_asset(savepath,
                                         file_name=metric_name,
                                         overwrite=False, step=this_step,
                                         metadata=None, copy_to_tmp=True)

    print('Logged metrics at: {}'.format(dirpath))
