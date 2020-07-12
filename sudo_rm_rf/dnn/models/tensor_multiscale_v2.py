"""!
@brief End-to-end tensor mask based network for
extremely efficient source separation

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


class ClassificationNet(nn.Module):
    def __init__(self, input_dimensions=None, num_classes=None,
                 model_type='simplest'):
        # Get some logits for a number of classes for the input
        # representation
        super(ClassificationNet, self).__init__()
        self.in_dims = input_dimensions
        assert len(self.in_dims) == 3, 'Input should be a 3d tensor.'
        assert all([isinstance(o, int) for o in self.in_dims])
        self.num_classes = num_classes

        if model_type == 'simplest':
            self.embedding_network = nn.ModuleList([
                nn.LeakyReLU(),
                # nn.LayerNorm([X * R, B, (64000 // (self.L-1))]),
                # nn.LayerNorm([X * R, B, 1]),
                nn.Conv2d(in_channels=self.in_dims[0],
                          out_channels=1,
                          kernel_size=1),
                nn.LeakyReLU(),
                # nn.LayerNorm([1, B, (64000 // (self.L - 1))]),
                nn.Conv2d(in_channels=1,
                          out_channels=1,
                          kernel_size=(self.in_dims[1], 1)),
                nn.LeakyReLU(),
                nn.LayerNorm([1, 1, self.in_dims[2]]),
            ])

            self.logits_layer = nn.Linear(self.in_dims[2],
                                          self.num_classes)
        elif model_type == '2d_separable_pooling':
            # Renormalize the input
            self.input_norm = nn.LayerNorm(self.in_dims[:2])
            self.embedding_network = nn.ModuleList(
                [nn.Conv2d(in_channels=self.in_dims[0],
                           out_channels=1,
                           kernel_size=1),] +
                # [SeparableConvBlock(self.in_dims[0] * 2 ** i,
                #                     self.in_dims[0] * 2 * 2 ** i)
                #  for i in range(8)]
                [SeparableConvBlock(self.in_dims[1],
                                    2 * self.in_dims[1],
                                    int(self.in_dims[2] / 2 ** i))
                 for i in range(int(np.log2(self.in_dims[2])) - 1)]
                # + [nn.AdaptiveMaxPool2d(1)]
            )

            self.logits_layer = nn.Linear(self.in_dims[1],
                                          self.num_classes)
        else:
            raise NotImplementedError('Classification Net type '
                                      'requested {} is not '
                                      'available.'.format(model_type))

    def forward(self, x):
        # Renormalize the input
        x = self.input_norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Get the source embedding
        for module in self.embedding_network:
            x = module(x)

        return self.logits_layer(x.view(x.shape[0], -1))



class EfficientMTDCN(nn.Module):

    class FlattenedDilatedConvBlock(nn.Module):
        def __init__(self, in_channels, embedding_channels,
                     kernel_size=3, dilation=1):
            super(EfficientMTDCN.FlattenedDilatedConvBlock, self).__init__()

            self.m = nn.ModuleList([
                nn.Conv1d(in_channels=in_channels,
                          out_channels=embedding_channels,
                          kernel_size=1),
                nn.PReLU(),
                GlobalLayerNorm(embedding_channels),
                nn.Conv1d(in_channels=embedding_channels,
                          out_channels=embedding_channels,
                          kernel_size=kernel_size,
                          padding=(dilation * (kernel_size - 1)) // 2,
                          dilation=dilation,
                          groups=embedding_channels // 4),
                nn.PReLU(),
                GlobalLayerNorm(embedding_channels),
                nn.Conv1d(in_channels=embedding_channels,
                          out_channels=in_channels,
                          kernel_size=1),
            ])

        def forward(self, x):
            y = x.clone()
            for layer in self.m:
                y = layer(y)
            return y + x

        def forward(self, x):
            y = x.clone()
            for layer in self.m:
                y = layer(y)

            return y + x

    def __init__(self, N_in, L_in, N_out, L_out,
                 B=256, H=512, P=3, X=8, R=3, S=2):
        super(EfficientMTDCN, self).__init__()

        self.S, self.B, self.H, self.P = S, B, H, P
        self.X, self.R = X, R
        # For every timescale pair
        self.L_in, self.N_in = L_in, N_in
        self.L_out, self.N_out = L_out, N_out

        # assert int((self.L_in - 1) / (self.L_out - 1)) == int(
        #     self.N_in / self.N_out)
        self.n_scales = (self.L_in - 1) // (self.L_out - 1)

        # List of front-ends
        self.front_end = nn.ModuleList([
                nn.Conv1d(in_channels=1, out_channels=N_in,
                          kernel_size=L_in, stride=L_in // 2,
                          padding=L_in // 2),
                nn.ReLU(),
                GlobalLayerNorm(N_in),
            ])


        self.sm = nn.ModuleList([
            EfficientMTDCN.FlattenedDilatedConvBlock(
                N_in,
                2 * N_in,
                kernel_size=P, dilation=2**d)
            for _ in range(R) for d in range(X)])

        self.mask_estimator = nn.Conv2d(
            in_channels=1,
            out_channels=S,
            kernel_size=(self.N_out + 1, 1),
            padding=(self.N_out // 2, 0))

        self.mask_norm = GlobalLayerNorm(self.N_out)

        # Higher analysis decoder only
        self.best_decoder = nn.ConvTranspose1d(
                            in_channels=S * self.N_out,
                            out_channels=S,
                            output_padding=(self.L_out // 2) - 1,
                            kernel_size=self.L_in,
                            stride=self.L_out // 2,
                            padding=self.L_in // 2,
                            groups=S)

    # Forward pass
    def forward(self, x):
        # Get the encoded tensor by gathering all
        encoded_tensors = []
        for fe in self.front_end:
            x = fe(x)

        s = x.clone()
        # s = s.view(s.shape[0], self.N_out, -1)

        # Get the embedding multiscale tensor
        for block in self.sm:
            x = block(x)
        #
        # # Convert the flattened tensor back to its shape for
        # # separated timescales
        # x = x.view(x.shape[0], self.n_scales, self.min_basis, -1)
        #
        # # Reconstruct all sources by estimating masks separately for
        # # each timescale
        # estimated_sources = []
        # for j, be in enumerate(self.back_ends):
        #     mask = x[:, j].view(x.shape[0], self.Ns[j], -1).unsqueeze(1)
        #     mask = self.mask_norms[j](mask)
        #     mask = self.mask_estimators[j](mask)
        #     mask = nn.functional.relu(mask)
        #     mask = nn.functional.softmax(mask, dim=1)
        #     for module in be:
        #         estimated_sources.append(
        #             module((mask * encoded_tensors[j].unsqueeze(1)).view(
        #                     x.shape[0], self.S * self.Ns[j], -1)))
        #
        # # Back end
        # return torch.stack(estimated_sources, dim=1).sum(dim=1)
        # # return self.be(x)

        # Get only the best decoder
        # print(x.shape)
        x = self.mask_norm(x)
        # print(x.shape)
        x = self.mask_estimator(x.unsqueeze(1))
        # print(x.shape)
        x = nn.functional.relu(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.view(x.shape[0], self.S, self.N_out, -1) * s.unsqueeze(1)
        x = torch.nn.functional.interpolate(
            x, size=(self.N_out, x.shape[-1] * self.n_scales),
            mode='bilinear')
        return self.best_decoder(x.view(x.shape[0], self.S *
                                        self.N_out,
                                        -1))


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
    X, R, L, B = 8, 4, 21, 256
    model = EfficientMTDCN(
        B=B,
        P=3,
        H=512,
        R=R,
        X=X,
        S=2,
        N_in=256,
        N_out=256,
        L_in=81,
        L_out=41)
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
