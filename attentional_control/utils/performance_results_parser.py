"""!
@brief Performance results measurements parser

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys


def parse_simple_results_file(path):
    filename = os.path.basename(path)
    basic_info = filename.split('_')
    result_dic = {
        'model_name': '_'.join(basic_info[:-5]),
        'batch_size': int(basic_info[-3]),
        'input_samples': int(basic_info[-1]),
        'device': basic_info[-5]
    }
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        info = line.split()
        if 'Trainable Parameters (millions):' in line:
            result_dic['Parameters (millions)'] = float(info[-1])
        elif 'GMACS:' in line:
            result_dic['Forward GMACS'] = float(info[-1])
        elif 'Elapsed Time Forward cpu:' in line:
            result_dic['Forward time (sec)'] = float(info[-2])
        elif 'Elapsed Time Forward gpu:' in line:
            result_dic['Forward time (sec)'] = float(info[-2])
        elif 'Peak GPU memory on Forward pass usage:' in line:
            result_dic['Forward memory (GB)'] = float(info[-2])
        elif 'Elapsed Time Backward gpu:' in line:
            result_dic['Backward time (sec)'] = float(info[-2])
        elif 'Peak GPU memory on Backward pass usage:' in line:
            result_dic['Backward memory (GB)'] = float(info[-2])
    return result_dic


def parse_cpuram_results_file(path):
    filename = os.path.basename(path)
    basic_info = filename.split('_')

    if basic_info[0].startswith('forward'):
        mode = 'Forward'
    elif basic_info[0].startswith('backward'):
        mode = 'Backward'
    else:
        raise NotImplementedError('Mode: {} is not valid for file in: {}'
                                  ''.format(basic_info[0], path))

    basic_info[0] = basic_info[0][len(mode)+len('CPURAM'):]

    result_dic = {
        'model_name': '_'.join(basic_info[:-5]),
        'batch_size': int(basic_info[-3]),
        'input_samples': int(basic_info[-1]),
        'device': basic_info[-5]
    }
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        info = line.split()
        if 'Maximum resident set size (kbytes):' in line:
            result_dic[mode + ' CPU RAM (GB)'] = float(info[-1]) / 10**6
    return result_dic


def gather_results_for_available_models(results_dir):
    filenames = os.listdir(results_dir)
    models = ['baseline_dprnn', 'baseline_demucs', 'baseline_twostep',
              'sudormrf_R16', 'sudormrf_R8', 'sudormrf_R4',
              'baseline_original_convtasnet']
    final_dic = dict([(m, {'cpu': {}, 'gpu': {}}) for m in models])

    for model in models:
        relevant_files = [f for f in filenames if model in f]
        for file in relevant_files:
            path = os.path.join(results_dir, file)
            if file.startswith('forward') or file.startswith('backward'):
                result_dic = parse_cpuram_results_file(path)
            else:
                result_dic = parse_simple_results_file(path)

            final_dic[model][result_dic['device']].update(result_dic)
    return final_dic


if __name__ == "__main__":
    out_dir = '/home/thymios/projects/attentional_control/performance_outputs'
    results_dic = gather_results_for_available_models(out_dir)
    # filikkename = 'sudormrf_R4_gpu_bs_1_samples_8000'
    # path = os.path.join(out_dir, filename)
    # result_dic = parse_simple_results_file(path)
    #
    # cpu_results_path = os.path.join(out_dir,
    #                                 'forwardCPURAMsudormrf_R4_gpu_bs_1_samples_8000')
    # cpu_results_dic = parse_cpuram_results_file(cpu_results_path)
    from pprint import pprint
    pprint(results_dic)
