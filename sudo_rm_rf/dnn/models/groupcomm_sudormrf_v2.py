"""!
@brief SuDO-RM-RF model with group communication block copied and modified from:
https://github.com/yluo42/GC3/blob/main/utility/GroupComm.py

The implementation refers to the GroupComm variation of the SuDo-RM-RF model
presented in (aka SuDoRM-RF++ GC):
Tzinis, E., Wang, Z., Jiang, X., and Smaragdis, P.,
“Compute and memory efficient universal sound source separation.”
In Journal of Signal Processing Systems, 2021 (to appear)
https://arxiv.org/pdf/2103.02644.pdf

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import math


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    '''
    This class defines the convolution layer with normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    '''
    This class defines a normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: number of output channels
        '''
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class UConvBlock(nn.Module):
    '''
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,
                                    stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,
                                           stride=1, groups=in_channels, d=1))

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=2*stride + 1,
                                               stride=stride,
                                               groups=in_channels, d=1))
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2,
                                               # align_corners=True,
                                               # mode='bicubic'
                                               )
        self.final_norm = NormAct(in_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth-1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.final_norm(output[-1])

        return self.res_conv(expanded) + residual


class GroupCommSudoRmRf(nn.Module):
    def __init__(self,
                 in_audio_channels=1,
                 out_channels=256,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=5,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2,
                 group_size=16):
        super(GroupCommSudoRmRf, self).__init__()

        # Number of sources to produce
        self.in_audio_channels = in_audio_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        assert self.enc_kernel_size % 2, (
            'Be mindful to signal processing and choose an odd number for '
            'your filter size, since the hop size is going to be an even '
            'number.')
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
                       self.enc_kernel_size // 2,
                       2 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(in_channels=in_audio_channels,
                                 out_channels=enc_num_basis,
                                 kernel_size=enc_kernel_size,
                                 stride=enc_kernel_size // 2,
                                 padding=enc_kernel_size // 2,
                                 bias=False)
        torch.nn.init.xavier_uniform(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=enc_num_basis,
            out_channels=out_channels,
            kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(*[
            GC_UConvBlock(out_channels=out_channels,
                          in_channels=in_channels,
                          upsampling_depth=upsampling_depth,
                          num_group=group_size)
            for _ in range(num_blocks)])

        mask_conv = nn.Conv1d(out_channels,
                              num_sources * enc_num_basis * in_audio_channels,
                              1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources * in_audio_channels,
            out_channels=num_sources * in_audio_channels,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=1, bias=False)
        torch.nn.init.xavier_uniform(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()
    # Forward pass
    def forward(self, input_wav):
        # Front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0],
                   self.num_sources * self.in_audio_channels,
                   self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        return self.remove_trailing_zeros(estimated_waveforms, input_wav)

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32)
            padded_x[..., :x.shape[-1]] = x
            return padded_x.to(x.device)
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


# transform-average-concatenate (TAC)
class TAC(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TAC, self).__init__()

        self.TAC_input = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.PReLU())
        self.TAC_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.PReLU())
        self.TAC_output = nn.Sequential(nn.Linear(hidden_size * 2, input_size),
                                        nn.PReLU())
        # self.TAC_norm = nn.GroupNorm(1, input_size)
        self.TAC_norm = GlobLN(input_size)

    def forward(self, input):
        # input shape: batch, group, N, seq_length

        batch_size, G, N, T = input.shape
        output = input

        # transform group_input = output  # B, G, N, T
        group_input = output.permute(0, 3, 1, 2).contiguous().view(-1,
                                                                   N)  # B*T*G, N
        group_output = self.TAC_input(group_input).view(batch_size, T, G,
                                                        -1)  # B, T, G, H

        # mean pooling
        group_mean = group_output.mean(2).view(batch_size * T, -1)  # B*T, H

        # concate
        group_output = group_output.view(batch_size * T, G, -1)  # B*T, G, H
        group_mean = self.TAC_mean(group_mean).unsqueeze(1).expand_as(
            group_output).contiguous()  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H
        group_output = self.TAC_output(
            group_output.view(-1, group_output.shape[-1]))  # B*T*G, N
        group_output = group_output.view(batch_size, T, G, -1).permute(0, 2, 3,
                                                                       1).contiguous()  # B, G, N, T
        group_output = self.TAC_norm(
            group_output.view(batch_size * G, N, T))  # B*G, N, T
        output = output + group_output.view(input.shape)

        return output


# GroupComm-UBlock
class GC_UConvBlock(nn.Module):
    '''
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    '''

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4,
                 num_group=16):
        super(GC_UConvBlock, self).__init__()

        self.num_group = num_group
        self.TAC = TAC(out_channels // num_group, out_channels * 3 // num_group)
        self.UBlock = UConvBlock(out_channels // num_group,
                                 in_channels // num_group,
                                 upsampling_depth=upsampling_depth)

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''
        batch_size, N, L = x.shape

        # TAC across groups
        output = self.TAC(x.view(batch_size, self.num_group, -1, L)).view(
            batch_size * self.num_group, -1, L)
        # UBlock for each group
        output = self.UBlock(output)

        return output.view(batch_size, N, L)


if __name__ == "__main__":
    in_audio_channels = 2
    fs = 16000
    model = GroupCommSudoRmRf(out_channels=256,
                              in_channels=512,
                              num_blocks=16,
                              upsampling_depth=7,
                              enc_kernel_size=91,
                              enc_num_basis=2048,
                              num_sources=4,
                              group_size=16)

    # fs = 8000
    timelength = 10.
    timesamples = int(fs * timelength)
    batch_size = 1
    dummy_input = torch.rand(batch_size, 1, timesamples)
    # model = model.cuda()
    # dummy_input = dummy_input.cuda()
    estimated_sources = model(dummy_input)
    print(estimated_sources.shape)
    assert estimated_sources.shape[-1] == dummy_input.shape[-1]




