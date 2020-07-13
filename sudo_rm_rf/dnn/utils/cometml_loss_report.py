"""!
@brief Library for experiment loss functionality

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import numpy as np


def report_losses_mean_and_std(losses_dict, experiment, tr_step, val_step):
    """Wrapper for cometml loss report functionality.

    Reports the mean and the std of each loss by inferring the train and the
    val string and it assigns it accordingly.

    Args:
        losses_dict: Python Dict with the following structure:
                     res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        experiment:  A cometml experiment object
        tr_step:     The step/epoch index for training
        val_step:     The step/epoch index for validation

    Returns:
        The updated losses_dict with the current mean and std
    """

    for l_name in losses_dict:
        values = losses_dict[l_name]['acc']
        mean_metric = np.mean(values)
        std_metric = np.std(values)

        if 'val' in l_name or 'test' in l_name:
            actual_name = l_name.replace('val_', '')
            with experiment.validate():
                experiment.log_metric(actual_name + '_mean',
                                      mean_metric,
                                      step=val_step)
                experiment.log_metric(actual_name + '_std',
                                      std_metric,
                                      step=val_step)
        elif 'tr' in l_name:
            actual_name = l_name.replace('tr_', '')
            with experiment.train():
                experiment.log_metric(actual_name + '_mean',
                                      mean_metric,
                                      step=tr_step)
                experiment.log_metric(actual_name + '_std',
                                      std_metric,
                                      step=tr_step)

        else:
            raise ValueError("tr or val or test must be in metric name <{}>."
                             "".format(l_name))

        losses_dict[l_name]['mean'] = mean_metric
        losses_dict[l_name]['std'] = std_metric

    return losses_dict
