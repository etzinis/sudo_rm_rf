"""!
@brief Effcient U-Net architecture

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob2
import datetime
import numpy as np


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.empty((1, channel_size, 1)))
        self.beta = nn.Parameter(torch.empty((1, channel_size, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2,
                                                keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1,
                                            keepdim=True).mean(dim=2,
                                                               keepdim=True)

        gLN_y = (self.gamma * (y - mean) /
                 torch.pow(var + 10e-8, 0.5) + self.beta)
        return gLN_y


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


class EfficientSeparableUNet(nn.Module):
    def __init__(self,
                 n_sources=2,
                 initial_basis=256,
                 basis_embedding=512,
                 depth=6,
                 glu=True,
                 upsample=False,
                 rescale=0.1,
                 initial_kernel=21,
                 computation_factor=2,
                 layer_stride=2,
                 layer_kernel=5,
                 growth=2.,
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
        self.n_sources=n_sources
        self.initial_basis=initial_basis
        self.basis_embedding=basis_embedding
        self.initial_kernel = initial_kernel
        self.layer_stride = layer_stride
        self.layer_kernel = layer_kernel
        self.context = context
        self.depth = depth
        self.upsample = upsample
        self.computation_factor = computation_factor

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.final = None
        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1


        self.front_end = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=self.initial_basis,
                      kernel_size=self.initial_kernel,
                      stride=self.initial_kernel // 2,
                      padding=self.initial_kernel // 2),
            nn.PReLU(),
            # GlobalLayerNorm(self.basis_embedding),
            nn.BatchNorm1d(self.initial_basis)
        ])

        in_channels = 1
        for index in range(depth):
            encode = []
            dilation = 2**index
            dilation_dec = 2**(depth-1-index)
            dilation = 1
            dilation_dec = 1
            encode += [
                # nn.Conv1d(in_channels=self.initial_basis,
                #           out_channels=self.basis_embedding,
                #           kernel_size=1),
                # nn.PReLU(),
                # GlobalLayerNorm(self.basis_embedding),
                nn.Conv1d(in_channels=self.basis_embedding,
                          out_channels=self.basis_embedding,
                          kernel_size=self.layer_kernel,
                          padding=(dilation * (self.layer_kernel-1)) // 2,
                          stride=self.layer_stride,
                          dilation=dilation,
                          groups=self.basis_embedding // self.computation_factor),
                nn.PReLU(),
                # GlobalLayerNorm(self.basis_embedding),
                nn.BatchNorm1d(self.basis_embedding),
                nn.Conv1d(in_channels=self.basis_embedding,
                          out_channels=self.initial_basis,
                          kernel_size=1),
                nn.PReLU(),
                # GlobalLayerNorm(self.basis_embedding),
                nn.BatchNorm1d(self.basis_embedding)]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                # nn.Conv1d(in_channels=self.initial_basis,
                #           out_channels=self.basis_embedding,
                #           kernel_size=1),
                # nn.PReLU(),
                # GlobalLayerNorm(self.basis_embedding),
                # nn.Conv1d(in_channels=self.basis_embedding,
                #           out_channels=self.basis_embedding,
                #           kernel_size=self.layer_kernel,
                #           padding=(dilation_dec * (self.layer_kernel - 1)) // 2,
                #           stride=self.layer_stride,
                #           dilation=dilation_dec,
                #           groups=self.basis_embedding // self.computation_factor),
                nn.ConvTranspose1d(
                    in_channels=self.basis_embedding,
                    out_channels=self.basis_embedding,
                    kernel_size=self.layer_kernel,
                    padding=(self.layer_kernel-1) // 2,
                    # output_padding=1,
                    output_padding=((self.layer_kernel - 1) // 2) - 1,
                    stride=self.layer_stride,
                    groups=self.basis_embedding // self.computation_factor),
                nn.PReLU(),
                # GlobalLayerNorm(self.basis_embedding),
                nn.BatchNorm1d(self.basis_embedding),
                nn.Conv1d(in_channels=self.basis_embedding,
                          out_channels=self.initial_basis,
                          kernel_size=1),
                nn.PReLU(),
                # GlobalLayerNorm(self.basis_embedding),
                nn.BatchNorm1d(self.basis_embedding)]

            self.decoder.insert(0, nn.Sequential(*decode))

        self.m = nn.Conv2d(in_channels=1,
                           out_channels=self.n_sources,
                           kernel_size=(self.initial_basis + 1, 1),
                           padding=(self.initial_basis // 2, 0))

        self.reshape_before_masks = nn.Conv1d(in_channels=2 * self.initial_basis,
                                              out_channels=self.initial_basis,
                                              kernel_size=1)

        # Back end
        self.back_end = nn.ConvTranspose1d(
            in_channels=self.initial_basis * self.n_sources,
            out_channels=self.n_sources,
            output_padding=(self.initial_kernel // 2) - 1,
            kernel_size=self.initial_kernel,
            stride=self.initial_kernel // 2,
            padding=self.initial_kernel // 2,
            groups=self.n_sources)
        self.ln_mask_in = GlobalLayerNorm(2 * self.initial_basis)

        self.lstm = BLSTM(self.initial_basis, layers=2)

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
        # First encode the mixture through the front end
        x = mix
        for fe in self.front_end:
            x = fe(x)
        saved = [x]
        # print('ENCODED MIXTURE REPRESENTATION')
        # print(x.shape)

        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        # print('BOTTLENECK REPRESENTATION')
        # print(x.shape)
        x = self.lstm(x)

        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)
            # print('DECODED REPRESENTATION')
            # print(x.shape)

        encoded_mixture = center_trim(saved.pop(-1), x)
        x = torch.cat([x, encoded_mixture], dim=1)
        x = self.ln_mask_in(x)
        x = self.reshape_before_masks(x)

        x = self.m(x.unsqueeze(1))
        x = nn.functional.relu(x)
        if self.n_sources == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)
        x = x * encoded_mixture.unsqueeze(1)

        return self.back_end(x.view(x.size(0), -1, x.size(-1)))


if __name__ == "__main__":
    import torch
    import os
    initial_basis = 512
    model = EfficientSeparableUNet(
                 n_sources=2,
                 initial_basis=initial_basis,
                 basis_embedding=initial_basis,
                 depth=7,
                 glu=True,
                 upsample=False,
                 rescale=0.1,
                 initial_kernel=11,
                 computation_factor=1,
                 layer_stride=2,
                 layer_kernel=5,
                 growth=2.,
                 context=3)
    # print('Try to fit the model in memory')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = model.cuda()
    # print(model)

    print('Testing Forward pass')
    dummy_input = torch.rand(1, 1, 32000).cuda()

    # import pdb; pdb.set_trace()

    import time
    now = time.time()
    pred_sources = model.forward(dummy_input)
    print(pred_sources.size())
    print('Elapsed: {}'.format(time.time()-now))

    try:
        from thop import profile
        macs, params = profile(model, inputs=(dummy_input,))
        print('MACS and params')
        print(round(macs / 10**6, 2), round(params / 10**6, 2))

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

