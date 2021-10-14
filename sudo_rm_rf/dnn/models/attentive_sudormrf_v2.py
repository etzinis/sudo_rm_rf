"""!
@brief Attentive SuDO-RM-RF model

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


class MHANormLayer(nn.Module):
    """Multi-head attention with residual addition and normalization."""
    def __init__(self, in_dim, att_dim=128,
                 num_heads=4, dropout=0.1,
                 max_len=5000):
        super(MHANormLayer, self).__init__()
        self.mha = nn.MultiheadAttention(
            att_dim, num_heads=num_heads, dropout=dropout,
            bias=True, add_bias_kv=False,
            add_zero_attn=False, kdim=None, vdim=None, batch_first=True,
            device=None, dtype=None)
        self.in_linear = nn.Linear(in_dim, att_dim)
        self.in_norm = GlobLN(att_dim)
        self.out_norm = GlobLN(att_dim)
        self.out_linear = nn.Linear(att_dim, in_dim)
        self.pos_enc = PositionalEncoding(
            d_model=att_dim, dropout=dropout, max_len=max_len)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.in_linear(x.transpose(1, 2))
        x = self.pos_enc(x)
        x = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.mha(query=x, key=x, value=x)[0]
        x = self.out_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.act(self.out_linear(x).transpose(1, 2))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=3200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class MHAttentionLayer(nn.Module):
    def __init__(self, emb_dim, d_model, n_heads, dropout=0.0):
        super(MHAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.Q_proj = nn.Linear(emb_dim, d_model * n_heads)
        self.K_proj = nn.Linear(emb_dim, d_model * n_heads)
        self.V_proj = nn.Linear(emb_dim, d_model * n_heads)
        self.O_proj = nn.Linear(d_model * n_heads, emb_dim)
        self.n_heads = n_heads
        self.d_model = d_model
        self.q_normalizer = 1. / math.sqrt(d_model)

    def forward(self, Q, K, V):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            Q: (batch_size, q_len, emb_dim)
            K: (batch_size, kv_len, emb_dim)
            V: (batch_size, kv_len, emb_dim)
        """
        bs, q_len, _ = Q.shape
        _, kv_len, _ = K.shape

        Q = self.q_normalizer * self.Q_proj(Q).view(
            bs, q_len, self.n_heads, -1)
        K = self.K_proj(K).view(bs, kv_len, self.n_heads, -1)
        V = self.V_proj(V).view(bs, kv_len, self.n_heads, -1)

        # Query-Key inner product per head
        QK = torch.einsum("nlhd,nshd->nhls", Q, K)

        # Compute the attention tensor over the values length
        A = self.dropout(torch.softmax(QK, dim=-1))

        # Compute the output tensor per head
        V = torch.einsum("nhls,nshd->nlhd", A, V)
        # V: (batch_size, q_len, n_heads, d_model)

        return self.O_proj(V.reshape(bs, q_len, -1))


class TransformerLayer(nn.Module):
    def __init__(self, emb_dim, d_model, n_heads,
                 dropout=0.1, max_len=5000):
        super(TransformerLayer, self).__init__()
        self.mha = MHAttentionLayer(
            emb_dim, d_model, n_heads, dropout=0.0)
        self.out_norm = GlobLN(emb_dim)
        self.out_mha_norm = GlobLN(emb_dim)
        self.ffn = ConvNormAct(emb_dim, emb_dim, 1, stride=1, groups=1)
        self.pos_enc = PositionalEncoding(
            d_model=emb_dim, dropout=dropout, max_len=max_len)

    def forward(self, x):
        # x has shape: (batch_size, n_channels, seq_len)
        x = self.pos_enc(x.transpose(1, 2))
        x = x + self.mha(Q=x, K=x, V=x)
        # x has shape: (batch_size, seq_len, n_channels)
        x = self.out_mha_norm(x.transpose(1, 2))
        # x has shape: (batch_size, n_channels, seq_len)

        # Apply the FFN layer
        return self.out_norm(self.ffn(x) + x)


class AttentiveUConvBlock(nn.Module):
    '''
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions. Moreover, it defines an attention layer which is used for
    better sequence modeling at the most downsampled level.
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4,
                 n_heads=4,
                 att_dims=256,
                 att_dropout=0.1):
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

        # Attention layer
        self.attention = TransformerLayer(
            in_channels, att_dims, n_heads,
            dropout=att_dropout, max_len=5000
        )

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
        for k in range(1, self.depth-1):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Perform the attention across time and accumulate
        att_in = self.spp_dw[self.depth-1](output[-1])
        output.append(self.attention(att_in))

        # Gather them now in reverse order
        for _ in range(self.depth-1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.final_norm(output[-1])

        return self.res_conv(expanded) + residual


class SuDORMRF(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 n_heads=4,
                 att_dims=256,
                 att_dropout=0.1,
                 num_sources=2):
        super(SuDORMRF, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
                       self.enc_kernel_size // 2,
                       2 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(in_channels=1, out_channels=enc_num_basis,
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
            AttentiveUConvBlock(out_channels=out_channels,
                                in_channels=in_channels,
                                upsampling_depth=upsampling_depth,
                                n_heads=4,
                                att_dims=256,
                                att_dropout=0.1)
            for _ in range(num_blocks)])

        mask_conv = nn.Conv1d(out_channels, num_sources * enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
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
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
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
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


if __name__ == "__main__":

    from time import time
    model = SuDORMRF(out_channels=256,
                     in_channels=512,
                     num_blocks=8,
                     upsampling_depth=5,
                     enc_kernel_size=21,
                     enc_num_basis=512,
                     n_heads=3,
                     att_dims=256,
                     att_dropout=0.1,
                     num_sources=4)

    dummy_input = torch.rand(3, 1, 32079)
    now = time()
    estimated_sources = model(dummy_input)
    print(estimated_sources.shape, time() - now)




