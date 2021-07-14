"""!
@brief Running an experiment with the improved version of SuDoRmRf on
universal source separation with multiple sources.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)

from __config__ import API_KEY
from comet_ml import Experiment, OfflineExperiment

import torch
from torch.nn import functional as F
from tqdm import tqdm
from pprint import pprint
import sudo_rm_rf.dnn.experiments.utils.improved_cmd_args_parser_v2 as parser
import sudo_rm_rf.dnn.experiments.utils.mixture_consistency \
    as mixture_consistency
import sudo_rm_rf.dnn.experiments.utils.dataset_setup as dataset_setup
import sudo_rm_rf.dnn.losses.sisdr as sisdr_lib
import sudo_rm_rf.dnn.losses.snr as snr_lib
import sudo_rm_rf.dnn.losses.norm as norm_lib
import sudo_rm_rf.dnn.models.improved_sudormrf as improved_sudormrf
import sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 as sudormrf_gc_v2
import sudo_rm_rf.dnn.models.causal_improved_sudormrf_v3 as \
    causal_improved_sudormrf
import sudo_rm_rf.dnn.models.sudormrf as initial_sudormrf
import sudo_rm_rf.dnn.utils.cometml_loss_report as cometml_report
import sudo_rm_rf.dnn.utils.cometml_log_audio as cometml_audio_logger
import sudo_rm_rf.dnn.utils.log_audio as offline_audio_logger

# torch.backends.cudnn.enabled = False
args = parser.get_args()
hparams = vars(args)
generators = dataset_setup.setup(hparams)
# Hardcode n_sources for all the experiments with musdb
assert hparams['n_channels'] == 1, 'Mono source separation is available for now'

audio_loggers = dict(
    [(n_src,
      cometml_audio_logger.AudioLogger(fs=hparams["fs"],
                                       bs=1,
                                       n_sources=n_src))
      for n_src in range(1, hparams['max_num_sources'] + 1)])

# offline_savedir = os.path.join('/home/thymios/offline_exps',
#                                hparams["project_name"],
#                                '_'.join(hparams['cometml_tags']))
# if not os.path.exists(offline_savedir):
#     os.makedirs(offline_savedir)
# audio_logger = offline_audio_logger.AudioLogger(dirpath=offline_savedir,
#     fs=hparams["fs"], bs=hparams["batch_size"], n_sources=4)

# Hardcode the test generator for each one of the number of sources
for n_src in range(hparams['min_num_sources'], hparams['max_num_sources']+1):
    for split_name in ['val', 'test']:
        loader = dataset_setup.create_loader_for_simple_dataset(
            dataset_name='FUSS',
            separation_task=hparams['separation_task'],
            data_split=split_name, sample_rate=hparams['fs'],
            n_channels=hparams['n_channels'], min_or_max=hparams['min_or_max'],
            zero_pad=hparams['zero_pad_audio'],
            timelegth=hparams['audio_timelength'],
            normalize_audio=hparams['normalize_audio'],
            n_samples=0, min_num_sources=n_src, max_num_sources=n_src)

        gen_name = '{}_{}_srcs'.format(split_name, n_src)
        generators[gen_name] = loader.get_generator(
            batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])

# experiment = OfflineExperiment(API_KEY, offline_directory=offline_savedir)
experiment = Experiment(API_KEY, project_name=hparams['project_name'])
experiment.log_parameters(hparams)
experiment_name = '_'.join(hparams['cometml_tags'])
for tag in hparams['cometml_tags']:
    experiment.add_tag(tag)
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

back_loss_tr_loss_name, back_loss_tr_loss = (
    'tr_back_loss_SNR',
    # norm_lib.L1(return_individual_results=False)
    # norm_lib.PermInvariantL1(n_sources=hparams["n_sources"],
    #                          weighted_norm=True)
    # 'tr_back_loss_SISDRi',
    snr_lib.PermInvariantSNRwithZeroRefs(
        n_sources=hparams["max_num_sources"],
        zero_mean=False,
        backward_loss=True,
        inactivity_threshold=-40.)
)

val_losses = {}
all_losses = []
for val_set in [x for x in generators if not x == 'train']:
    if generators[val_set] is None:
        continue

    n_actual_sources = int(val_set.split('_')[1])
    if n_actual_sources == 1:
        single_source = False
        improvement = False
        metric_name = 'SISDR'
        n_estimated_sources = 1
    else:
        single_source = False
        improvement = True
        n_estimated_sources = hparams['max_num_sources']
        metric_name = 'SISDRi'
    val_losses[val_set] = {}
    all_losses.append(val_set + '_{}'.format(metric_name))
    val_losses[val_set][val_set + '_{}'.format(metric_name)] = \
        sisdr_lib.StabilizedPermInvSISDRMetric(
            zero_mean=True,
            single_source=single_source,
            n_estimated_sources=n_estimated_sources,
            n_actual_sources=n_actual_sources,
            backward_loss=False,
            improvement=improvement,
            return_individual_results=True)
all_losses.append(back_loss_tr_loss_name)

if hparams['model_type'] == 'relu':
    model = improved_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                       in_channels=hparams['in_channels'],
                                       num_blocks=hparams['num_blocks'],
                                       upsampling_depth=hparams['upsampling_depth'],
                                       enc_kernel_size=hparams['enc_kernel_size'],
                                       enc_num_basis=hparams['enc_num_basis'],
                                       num_sources=hparams['max_num_sources'])
elif hparams['model_type'] == 'causal':
    model = causal_improved_sudormrf.CausalSuDORMRF(
        in_audio_channels=1,
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        num_sources=hparams['max_num_sources'])
elif hparams['model_type'] == 'softmax':
    model = initial_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                      in_channels=hparams['in_channels'],
                                      num_blocks=hparams['num_blocks'],
                                      upsampling_depth=hparams['upsampling_depth'],
                                      enc_kernel_size=hparams['enc_kernel_size'],
                                      enc_num_basis=hparams['enc_num_basis'],
                                      num_sources=hparams['max_num_sources'])
elif hparams['model_type'] == 'groupcomm_v2':
    model = sudormrf_gc_v2.GroupCommSudoRmRf(
        in_audio_channels=hparams['n_channels'],
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        num_sources=hparams['max_num_sources'],
        group_size=16)
else:
    raise ValueError('Invalid model: {}.'.format(hparams['model_type']))

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
experiment.log_parameter('Parameters', numparams)
print('Trainable Parameters: {}'.format(numparams))

model = torch.nn.DataParallel(model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer=opt, mode='max', factor=1. / hparams['divide_lr_by'],
#     patience=hparams['patience'], verbose=True)


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


def online_augment(clean_sources):
    # clean_sources: (batch, n_sources, time)
    # Online mixing over samples of the batch. (This might cause to get
    # mixtures from the same type of sound but it's highly improbable).
    # Keep the exact same SNR distribution with the initial mixtures.
    n_sources = clean_sources.shape[1]
    batch_size = clean_sources.shape[0]

    initial_biases = torch.mean(clean_sources, dim=-1, keepdim=True)
    initial_energies = torch.std(clean_sources, dim=-1, keepdim=True)

    augmented_wavs_l = []
    for i in range(n_sources):
        augmented_wavs_l.append(clean_sources[torch.randperm(batch_size), i])
    augmented_wavs = torch.stack(augmented_wavs_l, 1)
    # augmented_wavs = normalize_tensor_wav(augmented_wavs)
    # augmented_wavs = (augmented_wavs * initial_energies) + initial_biases
    augmented_wavs = augmented_wavs[:, torch.randperm(n_sources)]
    augmented_wavs *= (torch.rand(batch_size, n_sources).unsqueeze(-1) + 0.5)

    return augmented_wavs


tr_step = 0
val_step = 0
prev_epoch_val_loss = 0.
for i in range(hparams['n_epochs']):
    res_dic = {}
    for loss_name in all_losses:
        res_dic[loss_name] = {'mean': 0., 'std': 0., 'median': 0., 'acc': []}
    print("FUSS Sudo-RM-RF: {} - {} || Epoch: {}/{}".format(
        experiment.get_key(), experiment.get_tags(), i+1, hparams['n_epochs']))
    model.train()

    sum_loss = 0.
    train_tqdm_gen = tqdm(generators['train'], desc='Training')
    for cnt, data in enumerate(train_tqdm_gen):
        opt.zero_grad()
        # data shape: (batch, n_sources, time_samples)
        clean_wavs = online_augment(data)
        clean_wavs = clean_wavs.cuda()

        input_mixture = torch.sum(clean_wavs, -2, keepdim=True)
        # input_mixture = normalize_tensor_wav(input_mixture)

        input_mix_std = input_mixture.std(-1, keepdim=True)
        input_mix_mean = input_mixture.mean(-1, keepdim=True)
        input_mixture = (input_mixture - input_mix_mean) / (
                    input_mix_std + 1e-9)

        # input_mix_std = input_mixture.std(-1, keepdim=True)
        # input_mix_mean = input_mixture.mean(-1, keepdim=True)
        # input_mixture = (input_mixture - input_mix_mean) / (input_mix_std + 1e-9)
        # clean_wavs = normalize_tensor_wav(clean_wavs, std=input_mix_std)

        rec_sources_wavs = model(input_mixture)
        # rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
        rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs,
                                                     input_mixture)

        # l = back_loss_tr_loss(normalize_tensor_wav(rec_sources_wavs),
        #                       normalize_tensor_wav(clean_wavs))
        l = back_loss_tr_loss(rec_sources_wavs,
                              clean_wavs)
        l.backward()

        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           hparams['clip_grad_norm'])

        opt.step()
        sum_loss += l.detach().item()
        train_tqdm_gen.set_description(
            "Training, Running Avg Loss: {}".format(sum_loss / (cnt + 1)))

    if hparams['patience'] > 0:
        if tr_step % hparams['patience'] == 0:
            new_lr = (hparams['learning_rate']
                      / (hparams['divide_lr_by'] ** (tr_step // hparams['patience'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
    tr_step += 1

    for val_set in [x for x in generators if not x == 'train']:
        if generators[val_set] is not None:
            n_actual_sources = int(val_set.split('_')[1])
            model.eval()
            n_songs_written = 10
            with torch.no_grad():
                for data in tqdm(generators[val_set],
                                 desc='Validation on {}'.format(val_set)):
                    clean_wavs = data.cuda()
                    input_mixture = torch.sum(clean_wavs, -2, keepdim=True)
                    # input_mixture = normalize_tensor_wav(input_mixture)
                    input_mix_std = input_mixture.std(-1, keepdim=True)
                    input_mix_mean = input_mixture.mean(-1, keepdim=True)
                    input_mixture = (input_mixture - input_mix_mean) / (
                            input_mix_std + 1e-9)

                    rec_sources_wavs = model(input_mixture)
                    # rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
                    rec_sources_wavs = mixture_consistency.apply(
                        rec_sources_wavs,
                        input_mixture)

                    for loss_name, loss_func in val_losses[val_set].items():
                        # l, best_perm = loss_func(
                        #     normalize_tensor_wav(rec_sources_wavs),
                        #     normalize_tensor_wav(clean_wavs),
                        #     return_best_permutation=True)
                        l, best_perm = loss_func(
                            rec_sources_wavs,
                            clean_wavs,
                            return_best_permutation=True)
                        res_dic[loss_name]['acc'] += l.tolist()

            audio_loggers[n_actual_sources].log_batch(
                rec_sources_wavs[:, best_perm.long().cuda()][0, 0].unsqueeze(0),
                clean_wavs[0].unsqueeze(0),
                input_mixture[0].unsqueeze(0),
                experiment, step=val_step, tag=val_set)

    val_step += 1

    res_dic = cometml_report.report_losses_mean_and_std(res_dic,
                                                        experiment,
                                                        tr_step,
                                                        val_step)

    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)
