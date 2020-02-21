"""!
@brief Run an attentive multiscale tasnet with adding all predictions

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys

sys.path.append('../../../')
from __config__ import API_KEY

from comet_ml import Experiment

import torch
import numpy as np
from tqdm import tqdm
from pprint import pprint
import attentional_control.dnn.dataset_loader.torch_dataloader as dataloader
import attentional_control.dnn.experiments.utils.dataset_specific_params \
    as dataset_specific_params
import attentional_control.dnn.losses.sisdr as sisdr_lib
import attentional_control.dnn.utils.cometml_loss_report as cometml_report
import attentional_control.dnn.utils.metrics_logger as metrics_logger
import attentional_control.dnn.utils.log_audio as log_audio
import attentional_control.dnn.experiments.utils.cmd_args_parser as parser
import attentional_control.dnn.models.embed_enriched_tdcn as emb_tdcn
import attentional_control.dnn.experiments.utils.hparams_parser as \
    hparams_parser


args = parser.get_args()
hparams = hparams_parser.get_hparams_from_args(args)
dataset_specific_params.update_hparams(hparams)

if hparams["log_path"] is not None:
    audio_logger = log_audio.AudioLogger(hparams["log_path"],
                                         hparams["fs"],
                                         hparams["bs"],
                                         hparams["n_sources"])

experiment = Experiment(API_KEY, project_name=hparams["project_name"])
experiment.log_parameters(hparams)

experiment_name = '_'.join(hparams['tags'])
for tag in hparams['tags']:
    experiment.add_tag(tag)

if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

# define data loaders
train_gen, val_gen, tr_val_gen = dataset_specific_params.get_data_loaders(hparams)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([cad
                                               for cad in hparams['cuda_devs']])

sep_loss_tr_loss_name, sep_loss_tr_loss = (
    'tr_sep_loss_SISDRi',
    sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                 n_sources=hparams['n_sources'],
                                 zero_mean=True,
                                 backward_loss=True,
                                 improvement=True))

class_loss_tr_loss_name, class_loss_tr_loss = (
    'tr_class_loss_crossentropy',
    torch.nn.CrossEntropyLoss()
    # torch.nn.BCEWithLogitsLoss()
)
total_loss_tr_name = "_".join([sep_loss_tr_loss_name,
                               str(hparams['class_loss_weight']) + '*',
                               class_loss_tr_loss_name])

val_losses = dict([
    ('val_SISDRi', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                                n_sources=hparams['n_sources'],
                                                zero_mean=True,
                                                backward_loss=False,
                                                improvement=True,
                                                return_individual_results=True))
  ])
val_loss_name = 'val_SISDRi'

# tr_val_losses = dict([
#     ('tr_SISDRi', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
#                                                n_sources=hparams['n_sources'],
#                                                zero_mean=True,
#                                                backward_loss=False,
#                                                improvement=True,
#                                                return_individual_results=True))])


model = emb_tdcn.TDCN(
        B=hparams['B'],
        H=hparams['H'],
        P=hparams['P'],
        R=hparams['R'],
        X=hparams['X'],
        L=hparams['n_kernel'],
        N=hparams['n_basis'],
        S=hparams['n_sources'])

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
print('Trainable Parameters: {}'.format(numparams))
experiment.log_parameter('Parameters', numparams)

model = torch.nn.DataParallel(model).cuda()

if hparams['optimizer'] == 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
elif hparams['optimizer'] == 'radam':
    import pytorch_warmup as warmup
    opt = torch.optim.AdamW(model.parameters(),
                            lr=hparams['learning_rate'],
                            betas=(0.9, 0.999),
                            weight_decay=0.01)
    num_steps = len(train_gen) * hparams['n_epochs']
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=num_steps)
    # warmup_scheduler = warmup.UntunedLinearWarmup(opt)
    warmup_scheduler = warmup.RAdamWarmup(opt)
else:
    raise NotImplementedError('Optimizer: {} is not available!'.format(
        hparams['optimizer']))

all_losses = [total_loss_tr_name, sep_loss_tr_loss_name,
              class_loss_tr_loss_name] + \
             [k for k in sorted(val_losses.keys())] + \
             ['val_accuracy', 'val_crossentropy']
             # [k for k in sorted(tr_val_losses.keys())]

tr_step = 0
val_step = 0
for i in range(hparams['n_epochs']):
    res_dic = {}
    for loss_name in all_losses:
        res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
    print("Experiment: {} - {} || Epoch: {}/{}".format(experiment.get_key(),
                                                       experiment.get_tags(),
                                                       i+1,
                                                       hparams['n_epochs']))
    model.train()

    for data in tqdm(train_gen, desc='Training'):
        opt.zero_grad()
        m1wavs = data[0].unsqueeze(1).cuda()
        clean_wavs = data[1].cuda()
        gt_labels = data[2].long().cuda()

        # targets = torch.zeros((gt_labels.shape[0],
        #                        hparams['n_sources'],
        #                        hparams['n_classes']))
        # targets = targets.scatter_(hparams['n_sources'],
        #                            gt_labels.unsqueeze(-1), 1)
        # targets = targets.sum(1).cuda()

        # m1wavs = data[0].unsqueeze(1)
        # clean_wavs = data[1]
        # gt_labels = data[2]

        rec_sources_wavs, logits = model(m1wavs, return_logits=True)
        # rec_sources_wavs = model(m1wavs, return_logits=False)

        l_sep, best_permutations = sep_loss_tr_loss(
            rec_sources_wavs, clean_wavs,
            initial_mixtures=m1wavs,
            return_best_permutation=True)
        all_losses_buffer = [l_sep]

        # Adding classification losses
        class_losses_acc = 0.
        for s in range(hparams['n_sources']):
            l_class = class_loss_tr_loss(
                logits[:, s, :], gt_labels[s==best_permutations].long())

            all_losses_buffer.append((hparams['class_loss_weight'] /
                float(hparams['n_sources'])) * l_class)
            class_losses_acc += l_class.item() / float(hparams['n_sources'])

        # l_class = class_loss_tr_loss(logits, targets)
        # class_losses_acc += l_class.item()
        # all_losses_buffer.append((hparams['class_loss_weight'] /
        #                          float(hparams['n_sources'])) * l_class)

        total_loss = sum(all_losses_buffer)
        total_loss.backward()

        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           hparams['clip_grad_norm'])
        opt.step()
        if hparams['optimizer'] == 'radam':
            lr_scheduler.step()
            warmup_scheduler.dampen()
        res_dic[total_loss_tr_name]['acc'].append(total_loss.item())
        res_dic[sep_loss_tr_loss_name]['acc'].append(l_sep.item())
        res_dic[class_loss_tr_loss_name]['acc'].append(class_losses_acc)

    if hparams['reduce_lr_every'] > 0:
        if tr_step % hparams['reduce_lr_every'] == 0:
            new_lr = (hparams['learning_rate']
                      / (hparams['divide_lr_by'] ** (tr_step // hparams['reduce_lr_every'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
    tr_step += 1

    if val_gen is not None:
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_gen, desc='Validation'):
                m1wavs = data[0].unsqueeze(1).cuda()
                clean_wavs = data[1].cuda()
                gt_labels = data[2].long().cuda()

                # targets = torch.zeros((gt_labels.shape[0],
                #                        hparams['n_sources'],
                #                        hparams['n_classes']))
                # targets = targets.scatter_(hparams['n_sources'],
                #                            gt_labels.unsqueeze(-1), 1)
                # targets = targets.sum(1).cuda()

                rec_sources_wavs, logits = model(m1wavs,
                                                 return_logits=True)

                l_sep, best_permutations = sep_loss_tr_loss(
                    rec_sources_wavs, clean_wavs,
                    initial_mixtures=m1wavs,
                    return_best_permutation=True)
                all_losses_buffer = [l_sep]

                # Computing classification losses
                class_losses_acc = 0.
                for s in range(hparams['n_sources']):
                    true_ys = gt_labels[s == best_permutations].long()
                    l_class = class_loss_tr_loss(
                        logits[:, s, :], true_ys) / hparams['n_sources']
                    res_dic['val_crossentropy']['acc'] += [l_class.item()]

                    pred = logits[:, s, :].max(1, keepdim=True)[1]
                    correct = pred.eq(true_ys.view_as(pred)).tolist()
                    res_dic['val_accuracy']['acc'] += [correct]

                # l_class = class_loss_tr_loss(logits, targets)
                # res_dic['val_crossentropy']['acc'] += [l_class.item()]
                # pred = torch.topk(logits, 2, dim=1)[1]
                # predicted = torch.zeros((pred.shape[0],
                #                          hparams['n_sources'],
                #                          hparams['n_classes'])).cuda()
                # predicted = predicted.scatter_(hparams['n_sources'],
                #                                pred.unsqueeze(-1), 1)
                # predicted = predicted.sum(1).cuda()
                # correct = (targets * predicted).sum(-1)
                # correct = (correct / hparams['n_sources']).tolist()
                # res_dic['val_accuracy']['acc'] += [correct]

                for loss_name, loss_func in val_losses.items():
                    l = loss_func(rec_sources_wavs,
                                  clean_wavs,
                                  initial_mixtures=m1wavs)
                    res_dic[loss_name]['acc'] += l.tolist()

            if hparams["log_path"] is not None:
                audio_logger.log_batch(rec_sources_wavs,
                                       clean_wavs,
                                       m1wavs)
        val_step += 1

    # if tr_val_losses.values():
    #     model.eval()
    #     with torch.no_grad():
    #         for data in tqdm(tr_val_gen, desc='Train Validation'):
    #             m1wavs = data[0].unsqueeze(1).cuda()
    #             clean_wavs = data[-1].cuda()
    #
    #             rec_sources_wavs = model(m1wavs)
    #             for loss_name, loss_func in tr_val_losses.items():
    #                 l = loss_func(rec_sources_wavs,
    #                               clean_wavs,
    #                               initial_mixtures=m1wavs)
    #                 res_dic[loss_name]['acc'] += l.tolist()
    # if hparams["metrics_log_path"] is not None:
    #     metrics_logger.log_metrics(res_dic, hparams["metrics_log_path"],
    #                                tr_step, val_step)
    #
    res_dic = cometml_report.report_losses_mean_and_std(res_dic,
                                                        experiment,
                                                        tr_step,
                                                        val_step)

    # model_class.save_if_best(
    #     hparams['tn_mask_dir'], model.module, opt, tr_step,
    #     res_dic[back_loss_tr_loss_name]['mean'],
    #     res_dic[val_loss_name]['mean'], val_loss_name.replace("_", ""))
    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)
