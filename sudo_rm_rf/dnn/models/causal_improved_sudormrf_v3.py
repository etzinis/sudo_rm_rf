"""!
@brief Causal and simpler SuDO-RM-RF model

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import math

class ScaledWSConv1d(nn.Conv1d):
    """1D Conv layer with Scaled Weight Standardization."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0,
                 dilation=1, groups=1, bias=True, gain=False,
                 eps=1e-8):
        nn.Conv1d.__init__(self, in_channels, out_channels,
                           kernel_size, stride, padding, dilation,
                           groups, bias)
        self.causal_mask = torch.ones_like(self.weight)
        if kernel_size >= 3:
            future_samples = kernel_size // 2
            self.causal_mask[..., -future_samples:] = 0.

    def get_weight(self):
        return self.weight * self.causal_mask.to(self.weight.device)

    def forward(self, x):
        return nn.functional.conv1d(
            x, self.get_weight(), self.bias,
            self.stride, self.padding, self.dilation, self.groups)

class ConvAct(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = ScaledWSConv1d(nIn, nOut, kSize, stride=stride,
                                   padding=((kSize - 1) // 2), groups=groups)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        return self.act(output)


class UConvBlock(nn.Module):
    '''
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4,
                 alpha=1.,
                 beta=1.,):
        super().__init__()
        self.beta, self.alpha = beta, alpha
        self.skipinit_gain = nn.Parameter(torch.zeros(()))
        self.proj_1x1 = ConvAct(out_channels, in_channels, 1,
                                stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(ConvAct(in_channels, in_channels, kSize=21,
                                   stride=1, groups=in_channels))

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(ConvAct(in_channels, in_channels,
                                       kSize=21,
                                       stride=stride,
                                       groups=in_channels))
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2,
                                               # align_corners=True,
                                               # mode='bicubic'
                                               )
        self.res_conv = ScaledWSConv1d(in_channels, out_channels, 1)

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x / self.beta)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth-1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        return self.res_conv(output[-1]) * self.skipinit_gain * self.alpha + residual
        # return self.res_conv(output[-1]) + residual


class CausalSuDORMRF(nn.Module):
    def __init__(self,
                 in_audio_channels=1,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2):
        super(CausalSuDORMRF, self).__init__()

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
        self.encoder = ScaledWSConv1d(in_channels=in_audio_channels,
                                      out_channels=enc_num_basis,
                                      kernel_size=enc_kernel_size * 2 - 1,
                                      stride=enc_kernel_size // 2,
                                      padding=(enc_kernel_size * 2 - 1) // 2,
                                      bias=False)
        torch.nn.init.xavier_uniform(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.bottleneck = ScaledWSConv1d(
            in_channels=enc_num_basis,
            out_channels=out_channels,
            kernel_size=1)

        # Separation module
        uconv_layers = []
        expected_var = 1.0
        alpha = 1.
        for _ in range(num_blocks):
            beta = expected_var ** 0.5
            uconv_layers.append(
                UConvBlock(out_channels=out_channels,
                           in_channels=in_channels,
                           upsampling_depth=upsampling_depth,
                           alpha=alpha,
                           beta=beta))
            # expected_var += alpha ** 2
        self.sm = nn.Sequential(*uconv_layers)

        mask_conv = ScaledWSConv1d(
            out_channels, num_sources * enc_num_basis * in_audio_channels, 1)
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
        self.mask_nl_class = nn.PReLU()
    # Forward pass
    def forward(self, input_wav):
        # Front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # Split paths
        # s = x.clone()
        # Separation module
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0],
                   self.num_sources * self.in_audio_channels,
                   self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        # x = x * s.unsqueeze(1)
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


if __name__ == "__main__":
    in_audio_channels = 2
    model = CausalSuDORMRF(in_audio_channels=in_audio_channels,
                           out_channels=256,
                           in_channels=512,
                           num_blocks=4,
                           upsampling_depth=5,
                           enc_kernel_size=21,
                           enc_num_basis=512,
                           num_sources=2)

    fs = 44100
    # fs = 8000
    timelength = 1.
    timesamples = int(fs * timelength)
    batch_size = 1
    dummy_input = torch.rand(batch_size, in_audio_channels, timesamples)
    # model = model.cuda()
    # dummy_input = dummy_input.cuda()
    estimated_sources = model(dummy_input)
    print(estimated_sources.shape)
    assert estimated_sources.shape[-1] == dummy_input.shape[-1]




