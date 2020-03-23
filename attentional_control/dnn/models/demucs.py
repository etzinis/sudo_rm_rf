"""!
@brief Demucs code copied from:
https://github.com/facebookresearch/demucs

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import math

import torch as th
from torch import nn
from torch.nn import functional as F
import functools


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def center_trim(tensor, reference):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


class BLSTM(nn.Module):
    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers,
                            hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def upsample(x, stride):
    """
    Linear upsampling, the output will be `stride` times longer.
    """
    batch, channels, time = x.size()
    weight = th.arange(stride, device=x.device, dtype=th.float) / stride
    x = x.view(batch, channels, time, 1)
    out = x[..., :-1, :] * (1 - weight) + x[..., 1:, :] * weight
    return out.reshape(batch, channels, -1)


def downsample(x, stride):
    """
    Downsample x by decimation.
    """
    return x[:, :, ::stride]


class Demucs(nn.Module):
    @capture_init
    def __init__(self,
                 sources=4,
                 audio_channels=2,
                 channels=64,
                 depth=6,
                 rewrite=True,
                 glu=True,
                 upsample=False,
                 rescale=0.1,
                 kernel_size=8,
                 stride=4,
                 growth=2.,
                 lstm_layers=2,
                 context=3):
        """
        Args:
            sources (int): number of sources to separate
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            rewrite (bool): add 1x1 convolution to each encoder layer
                and a convolution to each decoder layer.
                For the decoder layer, `context` gives the kernel size.
            glu (bool): use glu instead of ReLU
            upsample (bool): use linear upsampling with convolutions
                Wave-U-Net style, instead of transposed convolutions
            rescale (int): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            lstm_layers (int): number of lstm layers, 0 = no lstm
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time
                steps.
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.upsample = upsample
        self.channels = channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.final = None
        if upsample:
            self.final = nn.Conv1d(channels + audio_channels, sources * audio_channels, 1)
            stride = 1

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        in_channels = audio_channels
        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size,
                                 stride,
                                 # padding=(kernel_size - 1)//2
                                 ),
                       nn.ReLU()]
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels,
                                     1,
                                     # padding=0
                                     ), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                if upsample:
                    out_channels = channels
                else:
                    out_channels = sources * audio_channels
            if rewrite:
                decode += [nn.Conv1d(channels, ch_scale * channels,
                                     context,
                                     # padding=(context - 1)//2
                                     ),
                           activation]
            if upsample:
                decode += [
                    nn.Conv1d(channels, out_channels, kernel_size,
                              # padding=(kernel_size - 1)//2,
                              stride=1),
                ]
            else:
                decode += [nn.ConvTranspose1d(
                    channels, out_channels, kernel_size, stride,
                    # padding=(kernel_size)//2-1,
                    # output_padding=stride // 2
                )
                           ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels

        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length when context = 1. If context > 1,
        the two signals can be center trimmed to match.

        For training, extracts should have a valid length.For evaluation
        on full tracks we recommend passing `pad = True` to :method:`forward`.
        """
        for _ in range(self.depth):
            if self.upsample:
                length = math.ceil(length / self.stride) + self.kernel_size - 1
            else:
                length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.depth):
            if self.upsample:
                length = length * self.stride + self.kernel_size - 1
            else:
                length = (length - 1) * self.stride + self.kernel_size

        return int(length)

    def forward(self, mix):
        # print('mix')
        # print(mix.shape)
        # Preprocess appropriately:
        x = F.pad(mix.unsqueeze(1), (7210, 7210))
        saved = [x]
        for encode in self.encoder:
            x = encode(x)
            # print('after encoder')
            # print(x.shape)
            saved.append(x)
            if self.upsample:
                x = downsample(x, self.stride)
                # print('downsample')
                # print(x.shape)
        if self.lstm:
            x = self.lstm(x)
            # print('lstm')
            # print(x.shape)
        for decode in self.decoder:
            if self.upsample:
                x = upsample(x, stride=self.stride)
                # print('upsample')
                # print(x.shape)
            skip = center_trim(saved.pop(-1), x)
            # print('skip')
            # print(skip.shape)
            x = x + skip
            # x = x + saved.pop(-1)
            x = decode(x)
            # print('after decoder')
            # print(x.shape)
        if self.final:
            skip = center_trim(saved.pop(-1), x)
            # print('after final trim')
            # print(skip.shape)
            x = th.cat([x, skip], dim=1)
            # x = th.cat([x, saved.pop(-1)], dim=1)
            x = self.final(x)
            # print('after final')
            # print(x.shape)

        x = x.view(x.size(0), self.sources, x.size(-1))
        return center_trim(x, mix)


if __name__ == "__main__":
    import torch
    import os
    model = Demucs(sources=2,
                     audio_channels=1,
                     channels=100,
                     depth=6,
                     rewrite=True,
                     glu=True,
                     upsample=False,
                     rescale=0.1,
                     kernel_size=8,
                     stride=4,
                     growth=2.,
                     lstm_layers=2,
                     context=3)
    # print('Try to fit the model in memory')
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = model.cuda()
    # print(model.summary())

    print('Testing Forward pass')
    # dummy_input = torch.rand(1, 1, 32000).cuda()
    dummy_input = torch.rand(1, 32000).cuda()

    # import pdb; pdb.set_trace()

    import time

    now = time.time()
    pred_sources = model.forward(dummy_input)
    print(pred_sources.size())
    print('Elapsed: {}'.format(time.time() - now))
    try:
        from thop import profile

        macs, params = profile(model, inputs=(dummy_input,))
        print('MACS and params')
        print(round(macs / 10 ** 6, 2), round(params / 10 ** 6, 2))

        from pytorch_memlab import profile


        @profile
        def work():
            pred_sources = model.forward(dummy_input)


        work()
    except:
        print('Could not find the profiler')

    numparams = 0
    for f in model.parameters():
        if f.requires_grad:
            numparams += f.numel()
    print('Trainable Parameters: {}'.format(numparams))
