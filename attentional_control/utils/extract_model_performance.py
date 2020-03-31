"""!
@brief Extract model performance measures using profilers and timing.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import argparse
import torch
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(root_dir)
import attentional_control.dnn.models.dprnn as dprnn
import attentional_control.dnn.models.demucs as demucs
import attentional_control.dnn.models.original_convtasnet as original_convtasnet
import attentional_control.dnn.models.simplified_tasnet as ptasent
import attentional_control.dnn.models.eetp_tdcn as eetptdcn
import attentional_control.dnn.losses.sisdr as sisdr_lib


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='Performance measures for various models.')
    parser.add_argument("--device", type=str,
                        help="The type of model you would like to use.",
                        default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument("--measure", type=str, nargs="+",
                        help="The type of measurements you would like to "
                             "extract by running this script.",
                        default='forward',
                        choices=['forward',
                                 'backward',
                                 'trainable_parameters',
                                 'macs',
                                 'memory'])
    parser.add_argument("--input_samples", type=int,
                        help="""Number of input time samples.""", default=8000)
    parser.add_argument("-cad", "--cuda_available_devices", type=str, nargs="+",
                        help="""A list of Cuda IDs that would be 
                        available for running this script.""",
                        default=['0'],
                        choices=['0', '1', '2', '3'])
    parser.add_argument("--model_type", type=str,
                        help="The type of model you would like to use.",
                        default='sudormrf_R4',
                        choices=['baseline_dprnn',
                                 'baseline_demucs',
                                 'baseline_twostep',
                                 'sudormrf_R16',
                                 'sudormrf_R8',
                                 'sudormrf_R4',
                                 'baseline_original_convtasnet'])
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                            Warning: Cannot be less than the number of 
                            the validation samples""", default=1)
    parser.add_argument("-r", "--repeats", type=int,
                        help="""The number of repetitions of the forward or
                        the backward pass""", default=1)
    return parser.parse_args()


def get_model(model_name):
    if model_name == 'baseline_dprnn':
        model_class = dprnn.FaSNet_base
        model = dprnn.FaSNet_base()
    elif model_name == 'baseline_original_convtasnet':
        model_class = original_convtasnet.TasNet
        model = original_convtasnet.TasNet()
    elif model_name == 'baseline_demucs':
        model_class = demucs.Demucs
        model = demucs.Demucs()
    elif model_name == 'baseline_twostep':
        model_class = ptasent.TDCN
        model = ptasent.TDCN(B=256, H=512, P=3, R=4, X=8, L=21, N=256, S=2)
    elif model_name == 'sudormrf_R16':
        model_class = eetptdcn.EETPTDCN
        model = eetptdcn.EETPTDCN(B=128, H=512, P=3, R=16, X=5, L=21, N=512, S=2)
    elif model_name == 'sudormrf_R8':
        model_class = eetptdcn.EETPTDCN
        model = eetptdcn.EETPTDCN(B=128, H=512, P=3, R=8, X=5, L=21, N=512, S=2)
    elif model_name == 'sudormrf_R4':
        model_class = eetptdcn.EETPTDCN
        model = eetptdcn.EETPTDCN(B=128, H=512, P=3, R=4, X=5, L=21, N=512, S=2)
    else:
        raise NotImplementedError(
            'Baseline model type: {} is not yet available.'.format(model_name))
    return model_class, model


def create_input_for_model(batch_size, input_samples, model_type):
    if model_type == 'baseline_demucs' or model_type == 'baseline_dprnn':
        dummy_input = torch.rand(batch_size, input_samples)
    else:
        dummy_input = torch.rand(batch_size, 1, input_samples)
    proper_input = torch.rand(batch_size, 1, input_samples)
    return dummy_input, proper_input


def create_targets(batch_size, input_samples, n_sources=2):
    return torch.rand(batch_size, n_sources, input_samples)


def count_parameters(model):
    numparams = 0
    for f in model.parameters():
        if f.requires_grad:
            numparams += f.numel()
    print('Trainable Parameters: {}'.format(round(numparams / 10**6, 3)))
    return numparams


def count_macs_for_forward(model, dummy_input):
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy_input,))
        print('MACS (millions): {}'.format(round(macs / 10**6, 2)))
    except:
        print('Could not find the profiler thop')
    return macs


def forward_pass(model, model_class,
                 repeats=1, mode='cpu', input_samples=8000, bs=4):
    now = time.time()
    for i in range(repeats):
        mixture, mixture_p = create_input_for_model(bs, input_samples,
                                                    model_class)
        if mode == 'gpu':
            mixture, mixture_p = mixture.cuda(), mixture_p.cuda()
        est_sources = model(mixture)
        print(est_sources.shape)
    avg_time = (time.time() - now) / repeats
    print('Elapsed Time Forward {}: {}'.format(mode, avg_time))


def backward_pass(model, model_class, input_samples=8000,
                  repeats=1, bs=4, n_sources=2, mode='cpu'):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    tr_loss = sisdr_lib.PermInvariantSISDR(batch_size=bs,
                                           n_sources=n_sources,
                                           zero_mean=True,
                                           backward_loss=True,
                                           improvement=True)

    total_time = 0.
    for i in range(repeats):
        mixture, mixture_p = create_input_for_model(bs, input_samples,
                                                    model_class)
        clean_wavs = create_targets(bs, input_samples, n_sources=n_sources)

        if mode == 'gpu':
            mixture, mixture_p = mixture.cuda(), mixture_p.cuda()
            clean_wavs = clean_wavs.cuda()

        now = time.time()
        opt.zero_grad()
        est_sources = model(mixture)

        l = tr_loss(est_sources, clean_wavs,
                    initial_mixtures=mixture_p.unsqueeze(1))

        l.backward()
        opt.step()
        total_time += time.time() - now
    avg_time = (total_time) / repeats
    print('Elapsed Time Backward {}: {}'.format(mode, avg_time))


def measure_gpu_memory(model, dummy_input, mode='forward'):
    try:
        from pytorch_memlab import profile
        @profile
        def work():
            pred_sources = model.forward(dummy_input)
        work()
    except Exception as e:
        print('Could not find the profiler pytorch_memlab')


if __name__ == "__main__":
    args = get_args()
    print('Selected Model Type: {}'.format(args.model_type))
    if 'memory' in args.measure and args.device == 'cpu':
        import tracemalloc
        tracemalloc.start()

    model_class, model = get_model(args.model_type)
    if args.device == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [cad for cad in args.cuda_available_devices])
        model = torch.nn.DataParallel(model).cuda()

    # 'forward',
    # 'backward',
    # 'trainable_parameters',
    # 'macs',
    # 'memory'

    if 'forward' in args.measure:
        forward_pass(model, args.model_type, repeats=args.repeats,
                     mode=args.device, input_samples=args.input_samples,
                     bs=args.batch_size)


    if 'memory' in args.measure and args.device == 'cpu':
        current, peak = tracemalloc.get_traced_memory()
        print('Current memory usage is {}GB; Peak was {}GB'
              ''.format(current / 10 ** 9, peak / 10 ** 9))
        tracemalloc.stop()
