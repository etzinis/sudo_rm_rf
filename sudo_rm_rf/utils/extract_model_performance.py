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
import sudo_rm_rf.dnn.models.dprnn as dprnn
import sudo_rm_rf.dnn.models.demucs as demucs
import sudo_rm_rf.dnn.models.original_convtasnet as original_convtasnet
import sudo_rm_rf.dnn.models.two_step_tdcn as ptasent
import sudo_rm_rf.dnn.models.sudormrf as sudormrf
import sudo_rm_rf.dnn.losses.sisdr as sisdr_lib


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
                                 'forward_macs',
                                 'memory_cpu',
                                 'memory_gpu'
                                 ])
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
    parser.add_argument("--n_sources", type=int,
                        help="""Number of sources in mixtures""", default=2)
    parser.add_argument("-r", "--repeats", type=int,
                        help="""The number of repetitions of the forward or
                        the backward pass""", default=1)
    parser.add_argument('--run_all', action='store_true',
                        help='Runs all measures for the selected device.',
                        default=False)
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
        model_class = sudormrf.SuDORMRF
        model = sudormrf.SuDORMRF(
            out_channels=128,
            in_channels=512,
            num_blocks=16,
            upsampling_depth=5,
            enc_kernel_size=21,
            enc_num_basis=512,
            num_sources=2)
    elif model_name == 'sudormrf_R8':
        model_class = sudormrf.SuDORMRF
        model = sudormrf.SuDORMRF(
            out_channels=128,
            in_channels=512,
            num_blocks=8,
            upsampling_depth=5,
            enc_kernel_size=21,
            enc_num_basis=512,
            num_sources=2)
    elif model_name == 'sudormrf_R4':
        model_class = sudormrf.SuDORMRF
        model = sudormrf.SuDORMRF(
            out_channels=128,
            in_channels=512,
            num_blocks=4,
            upsampling_depth=5,
            enc_kernel_size=21,
            enc_num_basis=512,
            num_sources=2)
    else:
        raise NotImplementedError(
            'Baseline model type: {} is not yet available.'.format(model_name))
    return model_class, model


def create_input_for_model(batch_size, input_samples, model_type):
    if model_type == 'baseline_demucs' or model_type == 'baseline_dprnn':
        dummy_input = torch.rand(batch_size, input_samples)
    else:
        dummy_input = torch.rand(batch_size, 1, input_samples)
    proper_input = torch.rand(batch_size, input_samples)
    return dummy_input, proper_input


def create_targets(batch_size, input_samples, n_sources=2):
    return torch.rand(batch_size, n_sources, input_samples)


def count_parameters(model):
    numparams = 0
    for f in model.parameters():
        if f.requires_grad:
            numparams += f.numel()
    print('Trainable Parameters (millions): {}'.format(
        round(numparams / 10**6, 3)))
    return numparams


def count_macs_for_forward(model, model_class, mode='cpu',
                           input_samples=8000, bs=4):
    try:
        from thop import profile
        mixture, mixture_p = create_input_for_model(bs, input_samples,
                                                    model_class)
        if mode == 'gpu':
            mixture, mixture_p = mixture.cuda(), mixture_p.cuda()
        macs, _ = profile(model, inputs=(mixture,))
        print('GMACS: {}'.format(round(macs / 10**9, 3)))
    except:
        print('Could not find the profiler thop')


def forward_pass(model, model_class,
                 repeats=1, mode='cpu', input_samples=8000, bs=4):
    total_time = 0.
    for i in range(repeats):
        mixture, mixture_p = create_input_for_model(bs, input_samples,
                                                    model_class)
        if mode == 'gpu':
            mixture, mixture_p = mixture.cuda(), mixture_p.cuda()
        now = time.time()
        est_sources = model(mixture)
        total_time += time.time() - now
    avg_time = total_time / repeats
    print('Elapsed Time Forward {}: {} sec'.format(mode, avg_time))


def backward_pass(model, model_class, input_samples=8000,
                  repeats=1, bs=4, n_sources=2, mode='cpu'):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    tr_loss = sisdr_lib.PermInvariantSISDR(batch_size=bs,
                                           n_sources=n_sources,
                                           zero_mean=True,
                                           backward_loss=True,
                                           improvement=True)

    mixture, mixture_p = create_input_for_model(bs, input_samples,
                                                model_class)
    clean_wavs = create_targets(bs, input_samples, n_sources=n_sources)
    if mode == 'gpu':
        mixture, mixture_p = mixture.cuda(), mixture_p.cuda()
        clean_wavs = clean_wavs.cuda()

    total_time = 0.
    for i in range(repeats):
        now = time.time()
        opt.zero_grad()
        est_sources = model(mixture)

        l = tr_loss(est_sources, clean_wavs,
                    initial_mixtures=mixture_p.unsqueeze(1))

        l.backward()
        opt.step()
        total_time += time.time() - now
    avg_time = (total_time) / repeats
    print('Elapsed Time Backward {}: {} sec'.format(mode, avg_time))


def measure_gpu_memory(model, model_class, mode='forward',
                       input_samples=8000, device='cpu',
                       repeats=1, bs=4, n_sources=2):
    try:
        from pytorch_memlab import profile

        @profile
        def work():
            if mode == 'forward':
                forward_pass(model, model_class,
                             repeats=repeats, mode=device,
                             input_samples=input_samples, bs=bs)
            elif mode == 'backward':
                backward_pass(model, model_class, repeats=repeats, mode=device,
                              input_samples=input_samples, bs=bs,
                              n_sources=n_sources)
            else:
                raise NotImplementedError('Mode: {} is not yet '
                                          'available'.format(mode))
        work()
    except Exception as e:
        print(e)
        print('Could not find the profiler pytorch_memlab')


def main_analyzer(args):
    print('=' * 20)
    print('Selected Device: {}'.format(args.device))
    print('Selected Model Type: {}'.format(args.model_type))
    print('Selected Batch Size: {}'.format(args.batch_size))
    print('Selected Input Samples: {}'.format(args.input_samples))
    print('*' * 20)
    if 'memory_cpu' in args.measure:
        import tracemalloc
        tracemalloc.start()

    model_class, model = get_model(args.model_type)
    if args.device == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [cad for cad in args.cuda_available_devices])
        if len(args.cuda_available_devices) > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    # measure_gpu_memory(model, args.model_type, repeats=args.repeats,
    #                    mode='forward',
    #                    device=args.device, input_samples=args.input_samples,
    #                    bs=args.batch_size, n_sources=args.n_sources)

    if 'trainable_parameters' in args.measure:
        count_parameters(model)

    if 'forward_macs' in args.measure:
        count_macs_for_forward(model, args.model_type, mode=args.device,
                               input_samples=args.input_samples,
                               bs=args.batch_size)

    if 'forward' in args.measure:
        forward_pass(model, args.model_type, repeats=args.repeats,
                     mode=args.device, input_samples=args.input_samples,
                     bs=args.batch_size)
        if 'memory_gpu' in args.measure:
            print('Peak GPU memory on Forward pass usage: {} GB'
                  ''.format(torch.cuda.max_memory_allocated() / 10 ** 9))

    if 'backward' in args.measure:
        backward_pass(model, args.model_type, repeats=args.repeats,
                      mode=args.device, input_samples=args.input_samples,
                      bs=args.batch_size, n_sources=args.n_sources)
        if 'memory_gpu' in args.measure:
            print('Peak GPU memory on Backward pass usage: {} GB'
                  ''.format(torch.cuda.max_memory_allocated() / 10 ** 9))
    print('=' * 20)
    time.sleep(2)


if __name__ == "__main__":
    args = get_args()
    if args.run_all:
        new_args = args
        new_args.measure = ['forward', 'trainable_parameters']
        if args.device == 'gpu':
            new_args.measure.append('memory_gpu')
            new_args.measure.append('backward')
        else:
            new_args.measure.append('forward_macs')
        main_analyzer(new_args)
    else:
        main_analyzer(args)



