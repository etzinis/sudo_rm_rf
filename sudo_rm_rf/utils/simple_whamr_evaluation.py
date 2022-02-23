"""
Simple evaluation on fixed datasets of WSJ0-2mix and WHAMR!

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""
import os, sys
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import IPython.display as ipd
from asteroid.metrics import get_metrics
from pprint import pprint
import time
import pickle
from tqdm import tqdm

torch.cuda.empty_cache()

# Get the pretrained models
print("Pre-trained models available:")
for model_name in os.listdir('../../pretrained_models'):
    print(model_name)

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

anechoic_model_p = '../../pretrained_models/GroupCom_Sudormrf_U8_Bases512_WSJ02mix.pt'
anechoic_model_p = '../../pretrained_models/Improved_Sudormrf_U16_Bases512_WSJ02mix.pt'
anechoic_model_p = '../../pretrained_models/Improved_Sudormrf_U36_Bases2048_WSJ02mix.pt'
noisy_reverberant_model_p = '../../pretrained_models/Improved_Sudormrf_U16_Bases2048_WHAMRexclmark.pt'
noisy_reverberant_model_p = '../../pretrained_models/Improved_Sudormrf_U36_Bases4096_WHAMRexclmark.pt'

# Load the appropriate class modules
sys.path.append("../../")
import sudo_rm_rf.dnn.models.improved_sudormrf as improved_sudormrf
import sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 as sudormrf_gc_v2
import sudo_rm_rf.dnn.models.sepformer as sepformer
from speechbrain.pretrained import SepformerSeparation as sep_former_separator
import sudo_rm_rf.dnn.experiments.utils.mixture_consistency as mixture_consistency

# get all files for wham or whamr!
whamr_test_folder_path = '/mnt/data/whamr/wav8k/min/tt'
wsj02mix_test_file_names = os.listdir(os.path.join(whamr_test_folder_path, 'mix_clean_anechoic'))
whamrexcl_test_file_names = os.listdir(os.path.join(whamr_test_folder_path, 'mix_both_reverb'))
wsj02mix_test_file_names = [os.path.join(whamr_test_folder_path, 'mix_clean_anechoic',name)
                            for name in wsj02mix_test_file_names]
whamrexcl_test_file_names = [os.path.join(whamr_test_folder_path, 'mix_both_reverb',name)
                             for name in wsj02mix_test_file_names]

def get_tensors_for_chosen_file(chosen_mixture_path):
    mixture, _ = torchaudio.load(chosen_mixture_path)
    chosen_filename = os.path.basename(chosen_mixture_path)
    ground_truth_sources = torch.tensor(np.array([
        torchaudio.load(os.path.join(whamr_test_folder_path,
                                     's1_anechoic', chosen_filename))[0].detach().numpy()[0],
        torchaudio.load(os.path.join(whamr_test_folder_path,
                                     's2_anechoic', chosen_filename))[0].detach().numpy()[0]
    ]))

    return mixture[:, :56000], ground_truth_sources[:, :56000]

chosen_file = '446o030h_0.13806_444c020w_-0.13806.wav'
chosen_mixture_path = os.path.join(whamr_test_folder_path, 'mix_clean_anechoic', chosen_file)
print(get_tensors_for_chosen_file(chosen_mixture_path)[0].shape,
get_tensors_for_chosen_file(chosen_mixture_path)[1].shape)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
models_to_eval = [
    {
        'model_path': '../../pretrained_models/GroupCom_Sudormrf_U8_Bases512_WSJ02mix.pt',
        'is_sudo_model': True,
        'test_dataset': "WSJ02mix",
    },
    # {
    #     'model_path': '../../pretrained_models/Improved_Sudormrf_U16_Bases512_WSJ02mix.pt',
    #     'is_sudo_model': True,
    #     'test_dataset': "WSJ02mix",
    # },
    # {
    #     'model_path': '../../pretrained_models/Improved_Sudormrf_U36_Bases2048_WSJ02mix.pt',
    #     'is_sudo_model': True,
    #     'test_dataset': "WSJ02mix",
    # },
    # {
    #     'model_path': '../../pretrained_models/Improved_Sudormrf_U16_Bases2048_WHAMRexclmark.pt',
    #     'is_sudo_model': True,
    #     'test_dataset': "WHAMR!",
    # },
    # {
    #     'model_path': '../../pretrained_models/Improved_Sudormrf_U36_Bases4096_WHAMRexclmark.pt',
    #     'is_sudo_model': True,
    #     'test_dataset': "WHAMR!",
    # },
    # {
    #     'model_path': None,
    #     'is_sudo_model': False,
    #     'test_dataset': "WSJ02mix",
    # }
]

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

def get_model(model_info, is_gpu=False):
    if model_info['model_path'] is None:
        model = sep_former_separator.from_hparams(source="speechbrain/sepformer-wsj02mix",
                                       savedir='../notebooks/pretrained_models/sepformer-wsj02mix',
                                       run_opts={"device":"cuda"})
        model_name = "Sepformer"
    else:
        model = torch.load(model_info['model_path'])
        model_name = model_info['model_path'].split('/')[-1]
    return model, model_name

results_dic = {}

for model_info in models_to_eval:
    model, model_name = get_model(model_info, is_gpu=True)
    model.cuda()
    print("======================")
    print(f"Evaluating model: {model_name}")

    results_dic[model_name] = {}

    if model_info['test_dataset'] == "WSJ02mix":
        mixture_paths = wsj02mix_test_file_names
    else:
        mixture_paths = whamrexcl_test_file_names

    for chosen_mixture_path in tqdm(mixture_paths):
        input_mix, gt_sources = get_tensors_for_chosen_file(chosen_mixture_path)
        input_mix = input_mix.cuda()
        if model_info['is_sudo_model']:
            input_mix_std = input_mix.std(-1, keepdim=True)
            input_mix_mean = input_mix.mean(-1, keepdim=True)
            input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)
            est_sources = model(input_mix.unsqueeze(1))

            if "Group" in model_info['model_path']:
                est_sources = mixture_consistency.apply(est_sources, input_mix.unsqueeze(1))

        else:
            est_sources = model(input_mix).permute(0, 2, 1)

        try:
            all_metrics_dic = get_metrics(
                input_mix.detach().cpu().numpy(),
                normalize_tensor_wav(gt_sources).detach().cpu().numpy(),
                normalize_tensor_wav(est_sources[0]).detach().cpu().numpy(),
                compute_permutation=True, sample_rate=8000, metrics_list='all')
        except Exception as e:
            print(e)
            continue

        for k, v in all_metrics_dic.items():
            if k not in results_dic[model_name]:
                results_dic[model_name][k] = [v]
            else:
                results_dic[model_name][k].append(v)
        if 'sisdri' in results_dic[model_name]:
            results_dic[model_name]['sisdri'].append(all_metrics_dic['si_sdr'] - all_metrics_dic['input_si_sdr'])
        else:
            results_dic[model_name]['sisdri'] = [all_metrics_dic['si_sdr'] - all_metrics_dic['input_si_sdr']]

        del est_sources, input_mix
    with open(f'../notebooks/{model_name}_sep_perf_models.pickle', 'wb') as handle:
        pickle.dump(results_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del model

pprint(results_dic)
