"""!
@brief Hparams parser for the experiments. Creates a dictionary of all the
command line configurations.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""


def get_hparams_from_args(args):
    hparams = {
        "train_dataset": args.train,
        "val_dataset": args.val,
        "train_val_dataset": args.train_val,
        "experiment_name": args.experiment_name,
        "project_name": args.project_name,
        "R": args.tasnet_R,
        "P": args.tasnet_P,
        "X": args.tasnet_X,
        "B": args.tasnet_B,
        "H": args.tasnet_H,
        "afe_reg": args.adaptive_fe_regularizer,
        "n_kernel": args.n_kernel,
        "n_basis": args.n_basis,
        "bs": args.batch_size,
        "n_jobs": args.n_jobs,
        "tr_get_top": args.n_train,
        "val_get_top": args.n_val,
        "cuda_devs": args.cuda_available_devices,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "tags": args.cometml_tags,
        "log_path": args.experiment_logs_path,
        'weighted_norm': args.weighted_norm,
        "metrics_log_path": args.metrics_logs_path,
        "datasets_priors": args.datasets_priors,
        "max_abs_snr": args.max_abs_snr,
        'selected_timelength': args.selected_timelength,
        'fixed_seed': args.fixed_seed,
        'model_type': args.model_type,
        'divide_lr_by': args.divide_lr_by,
        'reduce_lr_every': args.reduce_lr_every,
        "fs": args.fs,
        "optimizer": args.optimizer,
        "clip_grad_norm": args.clip_grad_norm,
        "class_loss_weight": args.class_loss_weight,
        "n_classes": args.n_classes,
        "log_audio": args.log_audio,
        "out_channels": args.out_channels,
        "in_channels": args.in_channels,
        "num_blocks": args.num_blocks,
        "upsampling_depth": args.upsampling_depth,
        "enc_kernel_size": args.enc_kernel_size,
        "enc_num_basis": args.enc_num_basis,
    }
    return hparams
