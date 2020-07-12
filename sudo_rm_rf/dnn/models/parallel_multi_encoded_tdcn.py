"""!
@brief Parallel multiple encoders on different scales for time domain
source separation.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob2
import datetime
import numpy as np


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super(SeparableConvBlock, self).__init__()

        self.modules_separable = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=1),
            nn.LeakyReLU(),
            GlobalLayerNorm(out_channels),
            nn.Conv1d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=3,
                      groups=out_channels),
            nn.LeakyReLU(),
            GlobalLayerNorm(out_channels),
            # nn.BatchNorm1d(H),
            nn.Conv1d(in_channels=out_channels,
                      out_channels=in_channels, kernel_size=1),
        ])

        # self.modules_separable = nn.ModuleList([
        #     nn.Conv1d(in_channels=in_channels,
        #               out_channels=out_channels, kernel_size=1),
        #     nn.LeakyReLU(),
        #     # GlobalLayerNorm(out_channels),
        #     nn.LayerNorm(time_dim),
        #     nn.Conv2d(in_channels=out_channels,
        #               out_channels=out_channels,
        #               kernel_size=(3, 3), padding=1,
        #               groups=out_channels),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(time_dim),
        #     # GlobalLayerNorm(out_channels),
        #     nn.Conv2d(in_channels=out_channels,
        #               out_channels=out_channels, kernel_size=(1, 1)),
        # ])

    def forward(self, x):
        y = x.clone()
        for layer in self.modules_separable:
            y = layer(y.squeeze())
        return F.avg_pool2d(y, kernel_size=(1, 2))


class MTDCN(nn.Module):

    class DilatedTensorConvBlock(nn.Module):
        def __init__(self, B, H, P, D, n_scales):
            super(MTDCN.DilatedTensorConvBlock, self).__init__()


            # self.m = nn.ModuleList([
            #     nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1),
            #     nn.PReLU(),
            #     GlobalLayerNorm(H),
            #     # nn.BatchNorm1d(H),
            #     nn.Conv1d(in_channels=H, out_channels=H, kernel_size=P,
            #               padding=(D * (P - 1)) // 2, dilation=D, groups=H),
            #     nn.PReLU(),
            #     GlobalLayerNorm(H),
            #     # nn.BatchNorm1d(H),
            # ])
            #
            # self.next_layer_dense = nn.Conv1d(in_channels=H,
            #                                   out_channels=B,
            #                                   kernel_size=1)

            self.m = nn.ModuleList([
                nn.Conv2d(in_channels=n_scales, out_channels=n_scales,
                          kernel_size=(B, P),
                          dilation=(1, D),
                          padding=(B // 2, (D * (P - 1)) // 2)),
                nn.PReLU(),
                nn.GroupNorm(n_scales, n_scales),
            ])

        def forward(self, x):
            y = x.clone()
            for layer in self.m:
                y = layer(y)

            return y + x

    def __init__(self, Ns=[], Ls=[], B=256, H=512, P=3, X=8, R=3, S=2):
        super(MTDCN, self).__init__()

        self.S, self.B, self.H, self.P = S, B, H, P
        self.X, self.R = X, R
        # For every timescale pair
        self.Ls, self.Ns = Ls, Ns

        assert len(self.Ls) == len(self.Ns), 'Number of basis and ' \
                                             'Kernel sizes lists must ' \
                                             'have the same length'
        self.feature_map_size = self.Ns[0] / (self.Ls[0] // 2)
        print([(N, L) for (N, L) in zip(self.Ns, self.Ls)])
        assert all([self.feature_map_size == N / (L // 2)
                    for (N, L) in zip(self.Ns, self.Ls)])
        assert sorted(Ns) == Ns
        assert sorted(Ls) == Ls

        self.min_basis, self.max_basis = Ns[0], Ns[-1]
        self.min_kernel_len, self.max_kernel_len = Ls[0], Ls[-1]
        self.n_scales = len(self.Ns)

        # List of front-ends
        self.front_ends = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(in_channels=1, out_channels=N,
                          kernel_size=L, stride=L // 2, padding=L // 2),
                nn.ReLU(),
                GlobalLayerNorm(N),
            ])
            for (N, L) in zip(self.Ns, self.Ls)])

        # # Norm before the rest, and apply one more dense layer
        # self.ln = GlobalLayerNorm(N)
        # # self.ln = nn.BatchNorm1d(N)
        # self.l1 = nn.Conv1d(in_channels=N, out_channels=B, kernel_size=1)
        #
        # Separation module
        self.sm = nn.ModuleList([
            MTDCN.DilatedTensorConvBlock(
                B=self.min_basis - 1, H=H, P=P, D=2 ** d,
                n_scales=self.n_scales)
            for _ in range(R) for d in range(X)])
        #
        # if B != N:
        #     # self.ln_bef_out_reshape = GlobalLayerNorm(B)
        #     self.reshape_before_masks = nn.Conv1d(in_channels=B * X * R,
        #                                           out_channels=N,
        #                                           kernel_size=1)
        #     # self.ln_bef_masks = nn.GlobalLayerNorm(S * N)
        #
        # Masks layer
        self.m = nn.Conv2d(in_channels=self.n_scales,
                           out_channels=S,
                           kernel_size=(self.max_basis + 1, 1),
                           padding=(self.max_basis - self.max_basis // 2, 0))
        # # self.ln_mask_in = nn.BatchNorm1d(self.N)
        # self.ln_mask_in = GlobalLayerNorm(self.N)
        #
        # # Back end
        self.be = nn.ConvTranspose1d(in_channels=S * self.min_basis,
                                     out_channels=S,
                                     output_padding=(self.min_kernel_len // 2) - 1,
                                     kernel_size=self.min_kernel_len,
                                     stride=self.min_kernel_len // 2,
                                     padding=self.min_kernel_len // 2,
                                     groups=S)

    # Forward pass
    def forward(self, x):
        # Get the encoded tensor by gathering all
        encoded_tensor = []
        for fe in self.front_ends:
            y = x.clone()
            for module in fe:
                y = module(y)
            encoded_tensor.append(y)

        encoded_tensor = [feature_map.view(feature_map.shape[0],
                                           self.min_basis, -1)
                          for feature_map in encoded_tensor]

        # print('GLENTI')
        # print(encoded_tensor[0].shape)
        # print(encoded_tensor[1].shape)
        x = torch.stack(encoded_tensor, dim=1)

        # Split paths
        s = x.clone()

        # Get the embedding multiscale tensor
        for block in self.sm:
            x = block(x)

        # print('aljsdfl;asdjf;lkasjd')
        # print(x.shape)

        x = self.m(x)
        # print(x.shape)
        x = nn.functional.relu(x)
        # print(x.shape)
        if self.S == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)
            # print(x.shape)
        x = x * s
        del s

        # Back end
        return self.be(x.view(x.shape[0], -1, x.shape[-1]))
        # return self.be(x)


    @classmethod
    def save(cls, model, path, optimizer, epoch,
             tr_loss=None, cv_loss=None):
        package = cls.serialize(model, optimizer, epoch,
                                tr_loss=tr_loss, cv_loss=cv_loss)
        torch.save(package, path)

    @classmethod
    def load(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(N=package['N'],
                    L=package['L'],
                    B=package['B'],
                    H=package['H'],
                    P=package['P'],
                    X=package['X'],
                    R=package['R'],
                    S=package['S'])
        model.load_state_dict(package['state_dict'])
        return model

    @classmethod
    def load_best_model(cls, models_dir, freq_res, sample_res):
        dir_id = 'mtdcn_L_{}_N_{}'.format(sample_res,
                                                    freq_res)
        dir_path = os.path.join(models_dir, dir_id)
        best_path = glob2.glob(dir_path + '/best_*')[0]
        return cls.load(best_path)

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            'N': model.N,
            'L': model.L,
            'B': model.B,
            'H': model.H,
            'P': model.P,
            'X': model.X,
            'R': model.R,
            'S': model.S,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

    @classmethod
    def encode_model_identifier(cls,
                                metric_name,
                                metric_value):
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%s")

        file_identifiers = [metric_name, str(metric_value)]
        model_identifier = "_".join(file_identifiers + [ts])

        return model_identifier

    @classmethod
    def decode_model_identifier(cls,
                                model_identifier):
        identifiers = model_identifier.split("_")
        ts = identifiers[-1].split('.pt')[0]
        [metric_name, metric_value] = identifiers[:-1]
        return metric_name, float(metric_value), ts

    @classmethod
    def encode_dir_name(cls, model):
        model_dir_name = 'mtdcn_L_{}_N_{}'.format(model.L, model.N)
        return model_dir_name

    @classmethod
    def get_best_checkpoint_path(cls, model_dir_path):
        best_paths = glob2.glob(model_dir_path + '/best_*')
        if best_paths:
            return best_paths[0]
        else:
            return None

    @classmethod
    def get_current_checkpoint_path(cls, model_dir_path):
        current_paths = glob2.glob(model_dir_path + '/current_*')
        if current_paths:
            return current_paths[0]
        else:
            return None

    @classmethod
    def save_if_best(cls, save_dir, model, optimizer, epoch,
                     tr_loss, cv_loss, cv_loss_name):

        model_dir_path = os.path.join(save_dir, cls.encode_dir_name(model))
        if not os.path.exists(model_dir_path):
            print("Creating non-existing model states directory... {}"
                  "".format(model_dir_path))
            os.makedirs(model_dir_path)

        current_path = cls.get_current_checkpoint_path(model_dir_path)
        models_to_remove = []
        if current_path is not None:
            models_to_remove = [current_path]
        best_path = cls.get_best_checkpoint_path(model_dir_path)
        file_id = cls.encode_model_identifier(cv_loss_name, cv_loss)

        if best_path is not None:
            best_fileid = os.path.basename(best_path)
            _, best_metric_value, _ = cls.decode_model_identifier(
                best_fileid.split('best_')[-1])
        else:
            best_metric_value = -99999999

        if float(cv_loss) > float(best_metric_value):
            if best_path is not None:
                models_to_remove.append(best_path)
            save_path = os.path.join(model_dir_path, 'best_' + file_id + '.pt')
            cls.save(model, save_path, optimizer, epoch,
                     tr_loss=tr_loss, cv_loss=cv_loss)

        save_path = os.path.join(model_dir_path, 'current_' + file_id + '.pt')
        cls.save(model, save_path, optimizer, epoch,
                 tr_loss=tr_loss, cv_loss=cv_loss)

        try:
            for model_path in models_to_remove:
                os.remove(model_path)
        except:
            print("Warning: Error in removing {} ...".format(current_path))


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


class CepstralNorm(nn.Module):
    """Cepstral Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(CepstralNorm, self).__init__()
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
        mean = y.mean(dim=2, keepdim=True)
        var = ((y - mean)**2).mean(dim=2, keepdim=True)

        gLN_y = (self.gamma * (y - mean) /
                 torch.pow(var + 10e-8, 0.5) + self.beta)
        return gLN_y


if __name__ == "__main__":
    X, R, L, B = 8, 3, 21, 256
    model = MTDCN(
        B=B,
        P=3,
        H=512,
        R=R,
        X=X,
        S=2,
        Ns=[64, 256],
        Ls=[21, 81])
    print(model)

    numparams = 0
    for f in model.parameters():
        if f.requires_grad:
            numparams += f.numel()
    print('Trainable Parameters: {}'.format(numparams))

    print('Testing Forward pass')
    dummy_input = torch.rand(4, 1, 3200)
    pred_sources = model.forward(dummy_input)

    print('Output size: {}'.format(pred_sources.size()))
    assert pred_sources.shape[0] == dummy_input.shape[0]
    assert pred_sources.shape[-1] == dummy_input.shape[-1]

    #
    # classification_net = ClassificationNet(
    #     input_dimensions=[X * R, B, (64000 // (L - 1))],
    #     num_classes=50,
    #     model_type='2d_separable_pooling')
    # numparams = 0
    # for f in classification_net.parameters():
    #     if f.requires_grad:
    #         numparams += f.numel()
    # print('Classification Parameters: {}'.format(numparams))
    #
    # logits = classification_net(torch.rand(4, X * R, B,
    #                                        (64000 // (L - 1))))
    # assert logits.shape[-1] == 50
