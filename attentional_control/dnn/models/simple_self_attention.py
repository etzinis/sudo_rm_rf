"""!
@brief Simple time-domain self-attention

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import TransformerEncoder, \
    TransformerEncoderLayer


class SelfAttentionEncoder(nn.Module):

    def __init__(self, input_timesamples=32000, n_head=8, n_layers=6,
                 dim_feedforward=2048,
                 n_basis=64, kernel_size=21, n_sources=2,
                 dropout=0.1, activation="relu"):
        super(SelfAttentionEncoder, self).__init__()
        
        self.N = n_basis
        self.L = kernel_size
        self.T = input_timesamples
        self.n_head = n_head
        self.n_layers = n_layers
        self.n_sources = n_sources
        self.drop_rate = dropout
        self.input_timesamples = input_timesamples
        self.timesteps = int(self.input_timesamples // (self.L // 2))
        # self.d_model = int(self.input_timesamples // (self.L // 2))
        self.d_model = self.N

        # Front end
        self.fe = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=n_basis,
                      kernel_size=kernel_size,
                      stride=kernel_size // 2,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.LayerNorm([self.N, self.timesteps])
            # nn.BatchNorm1d(n_basis),
        ])
        

        # Separation Module with Self-Attention
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                nhead=self.n_head,
                                                dim_feedforward=dim_feedforward,
                                                dropout=self.drop_rate,
                                                activation=activation)
        encoder_norm = nn.LayerNorm(self.d_model)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, self.n_layers, encoder_norm)

        # # Masks layer
        # # Hmmmm maybe a linear layer here would be better
        # self.reshape_before_masks = nn.Conv2d(in_channels=1,
        #                                       out_channels=n_sources,
        #                                       kernel_size=(n_basis + 1, 1),
        #                                       padding=(n_basis - n_basis // 2, 0))
        # self.reshape_before_masks = nn.Linear(n_basis, n_basis * n_sources)

        # Back end
        self.be = nn.ConvTranspose1d(in_channels=n_basis*n_sources,
                                     out_channels=n_sources,
                                     output_padding=(kernel_size//2)-1,
                                     kernel_size=kernel_size,
                                     stride=kernel_size // 2,
                                     padding=kernel_size // 2,
                                     groups=n_sources)
        # self.ln_mask_in = nn.BatchNorm1d(self.N)
        # self.ln_mask_in = GlobalLayerNorm(n_sources)
        # self.ln_mask_in = nn.LayerNorm([self.N, self.timesteps])

    def forward(self, x):
        # Front end
        for l in self.fe:
            x = l(x)

        # Split paths
        s = x.clone()

        # Separation module
        # x = self.ln(x)
        # x = self.l1(x)
        # print(x.shape)
        x = self.transformer_encoder(x.permute(2, 0, 1)).permute(1, 2, 0)

        # x = self.reshape_before_masks(x.unsqueeze(1))
        # x = self.reshape_before_masks(x.permute(0, 2, 1)).permute(0, 2, 1)
        # # x = x.view(-1, 1, self.n_sources * self.N, self.timesteps)
        # x = x.view(-1, self.n_sources, self.N, self.timesteps)

        # x = self.ln_mask_in(x)
        # x = nn.functional.relu(x)
        if self.n_sources == 1:
            x = torch.sigmoid(x)
        else:
            # x = nn.functional.softmax(x, dim=1)
            x = torch.sigmoid(x.unsqueeze(1))
            sec_mask = (1. - x)
            x = torch.cat((x, sec_mask), 1)
        x = x * s.unsqueeze(1)

        return self.be(x.view(x.shape[0], -1, x.shape[-1]))


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