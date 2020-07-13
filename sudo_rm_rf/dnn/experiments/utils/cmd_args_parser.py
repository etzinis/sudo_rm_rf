"""!
@brief Experiment Argument Parser

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import argparse


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='CometML Experiment Argument Parser')
    parser.add_argument("--train", type=str, nargs='+',
                        help="Training dataset",
                        default=None,
                        choices=['WSJ2MIX8K', 'WSJ2MIX8KPAD',
                                 'TIMITMF8K', 'WSJ2MIX8KNORMPAD',
                                 'AUGMENTED_WSJMIX', 'AUGMENTED_ESC50'])
    parser.add_argument("--val", type=str, nargs='+',
                        help="Validation dataset",
                        default=None,
                        choices=['WSJ2MIX8K', 'WSJ2MIX8KPAD',
                                 'TIMITMF8K', 'WSJ2MIX8KNORMPAD',
                                 'AUGMENTED_WSJMIX', 'AUGMENTED_ESC50'])
    parser.add_argument("--train_val", type=str, nargs='+',
                        help="Validating on the training dataset",
                        default=None,
                        choices=['WSJ2MIX8K', 'WSJ2MIX8KPAD',
                                 'TIMITMF8K', 'WSJ2MIX8KNORMPAD',
                                 'AUGMENTED_WSJMIX', 'AUGMENTED_ESC50'])

    # Logging experiments
    parser.add_argument("-elp", "--experiment_logs_path", type=str,
                        help="""Path for experiment's checkpoints.""",
                        default=None)
    parser.add_argument("-mlp", "--metrics_logs_path", type=str,
                        help="""Path for logging metrics.""",
                        default=None)
    parser.add_argument('--log_audio', action='store_true', default=False)

    parser.add_argument("--n_train", type=int,
                        help="""Reduce the number of training 
                            samples to this number.""", default=None)
    parser.add_argument("--n_val", type=int,
                        help="""Reduce the number of evaluation 
                            samples to this number.""", default=None)
    parser.add_argument("-ri", "--return_items", type=str, nargs='+',
                        help="""A list of elements that this 
                        dataloader should return. See available 
                        choices which are based on the saved data 
                        names which are available. There is no type 
                        checking in this return argument.""",
                        default=['mixture_wav', 'clean_sources_wavs'],
                        choices=['mixture_wav',
                                 'clean_sources_wavs',
                                 'mixture_wav_norm', 'wav', 'class_id',
                                 'clean_sources_wavs_norm'])
    parser.add_argument("-tags", "--cometml_tags", type=str,
                        nargs="+", help="""A list of tags for the cometml 
                        experiment.""",
                        default=[])
    parser.add_argument("--experiment_name", type=str,
                        help="""Name of current experiment""",
                        default=None)
    parser.add_argument("--project_name", type=str,
                        help="""Name of current experiment""",
                        default="first_wsj02mix")

    # Augmented Dataset parameters
    parser.add_argument("-priors", "--datasets_priors", type=float, nargs='+',
                        help="The prior probability of finding a sample from "
                             "each given dataset. The length of this list "
                             "must be equal to the number of dataset paths "
                             "given above. The sum of this list must add up "
                             "to 1.",
                        default=[1])
    parser.add_argument("-fs", type=float,
                        help="""Sampling rate of the audio.""", default=8000.)
    parser.add_argument("--selected_timelength", type=float,
                        help="""The timelength of the sources that you want 
                                to load in seconds.""",
                        default=4.)
    parser.add_argument("--max_abs_snr", type=float,
                        help="""The maximum absolute value of the SNR of 
                                the mixtures.""", default=2.5)
    parser.add_argument("--fixed_seed", type=int,
                        help="""Whether the dataset is going to be fixed (
                            e.g. test partitions should be always fixed) give 
                            the random seed. If seed is zero then it means that 
                            the dataset is not going to be fixed.""",
                        default=0)

    # device params
    parser.add_argument("-cad", "--cuda_available_devices", type=str,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                        available for running this experiment""",
                        default=['0'],
                        choices=['0', '1', '2', '3'])
    # Adaptive front-end parameters
    parser.add_argument("-afe_reg", "--adaptive_fe_regularizer", type=str,
                        help="""regularization on the trained basis.""",
                        default=None,
                        choices=['compositionality',
                                 'softmax',
                                 'binarized'])

    # model params
    parser.add_argument("--model_type", type=str,
                        help="The type of model you would like "
                             "to use. In tasnet models you need to "
                             "choose between simple or residual.",
                        default='simple',
                        choices=['simple', 'residual',
                                 'baseline_dprnn',
                                 'baseline_demucs',
                                 'eetp_tdcn',
                                 'eunet',
                                 'baseline_original_convtasnet'])
    parser.add_argument("-N", "--n_basis", type=int,
                        help="Dim of encoded representation",
                        default=256)
    parser.add_argument("-B", "--tasnet_B", type=int,
                        help="Number of dimensions in bottleneck.",
                        default=256)
    parser.add_argument("-H", "--tasnet_H", type=int,
                        help="Number of channels in convd.",
                        default=512)
    parser.add_argument("-L", "--n_kernel", type=int,
                        help="Length of encoding convolutional filters",
                        default=20)
    parser.add_argument("-P", "--tasnet_P", type=int,
                        help="Length of filters in convolution blocks",
                        default=3)
    parser.add_argument("-X", "--tasnet_X", type=int,
                        help="Number of convolutional blocks in each repeat",
                        default=8)
    parser.add_argument("-R", "--tasnet_R", type=int,
                        help="Number of repeats of TN Blocks",
                        default=4)
    parser.add_argument("-Sc", "--tasnet_Sc", type=int,
                        help="Number of dimensions of Skip Connection Dense "
                             "Layer",
                        default=128)
    parser.add_argument("-norm", "--norm_type", type=str,
                        help="""The type of the applied normalization layer.""",
                        default="gln", choices=['bn', 'gln'])
    parser.add_argument("-wn", "--weighted_norm", type=float,
                        help="""The exponent of weighting L1 in mask 
                        regression.""",
                        default=0.0)
    parser.add_argument("-version", "--tasnet_version", type=str,
                        help="""The type of Tasnet you want to run.""",
                        default="simplified", choices=['full', 'simplified'])
    # training params
    parser.add_argument("--class_loss_weight", type=float,
                        help="""The weight that classification loss 
                        is going to be added with the separation 
                        loss. 0 means that there would be no 
                        classification loss.""",
                        default=0.)
    parser.add_argument("--n_classes", type=int,
                        help="""The number of classes of sounds 
                        that will be used.""",
                        default=50)

    # ===============================================
    # Separation model (SuDO-RM-RF) params
    parser.add_argument("--out_channels", type=int,
                        help="The number of channels of the internal "
                             "representation outside the U-Blocks.",
                        default=128)
    parser.add_argument("--in_channels", type=int,
                        help="The number of channels of the internal "
                             "representation inside the U-Blocks.",
                        default=512)
    parser.add_argument("--num_blocks", type=int,
                        help="Number of the successive U-Blocks.",
                        default=16)
    parser.add_argument("--upsampling_depth", type=int,
                        help="Number of successive upsamplings and "
                             "effectively downsampling inside each U-Block. "
                             "The aggregation of all scales is performed by "
                             "addition.",
                        default=5)
    parser.add_argument("--enc_kernel_size", type=int,
                        help="The width of the encoder and decoder kernels.",
                        default=21)
    parser.add_argument("--enc_num_basis", type=int,
                        help="Number of the encoded basis representations.",
                        default=512)

    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                            Warning: Cannot be less than the number of 
                            the validation samples""", default=4)
    parser.add_argument("--n_jobs", type=int,
                        help="""The number of cpu workers for 
                            loading the data, etc.""", default=4)
    parser.add_argument("--n_epochs", type=int,
                        help="""The number of epochs that the 
                        experiment should run""", default=50)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-2)
    parser.add_argument("--divide_lr_by", type=float,
                        help="""The factor that the learning rate 
                        would be divided by""", default=1.)
    parser.add_argument("--reduce_lr_every", type=float,
                        help="""Reduce learning rate every how many 
                        training epochs? 0 means that the learning 
                        rate is not going to be divided by the 
                        specified factor.""",
                        default=0)
    parser.add_argument("--optimizer", type=str,
                        help="""The optimizer that you want to use""",
                        default="adam",
                        choices=['adam', 'radam'])
    parser.add_argument("--clip_grad_norm", type=float,
                        help="""The norm value which all gradients 
                        are going to be clipped, 0 means that all 
                        grads are not going to be clipped""",
                        default=0.)

    return parser.parse_args()
