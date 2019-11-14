"""!
@brief Library for saving metrics per sample and per epoch

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import numpy as np


def log_metrics(metrics_dict, dirpath, tr_step, val_step):
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
        if 'tr' in metric_name:
            filename = 'epoch_{}'.format(tr_step)
        else:
            filename = 'epoch_{}'.format(val_step)
        np.save(os.path.join(this_metric_folder, filename), values)
