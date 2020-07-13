"""!
@brief Two step TDCN model from:
https://github.com/etzinis/two_step_mask_learning/tree/master/two_step_mask_learning/dnn/models

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import os
import glob2
import datetime


class TDCN(nn.Module):

    # Simplified TCN layer
    class TCN(nn.Module):
        def __init__(self, B, H, P, D):
            super(TDCN.TCN, self).__init__()

            self.m = nn.ModuleList([
                nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1),
                nn.PReLU(),
                GlobalLayerNorm(H),
                # nn.BatchNorm1d(H),
                nn.Conv1d(in_channels=H, out_channels=H, kernel_size=P,
                          padding=(D * (P - 1)) // 2, dilation=D, groups=H),
                nn.PReLU(),
                GlobalLayerNorm(H),
                # nn.BatchNorm1d(H),
                nn.Conv1d(in_channels=H, out_channels=B, kernel_size=1),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l(y)
            return x + y

    # Set things up
    def __init__(self, N, L, B, H, P, X, R, S=1):
        super(TDCN, self).__init__()

        # Number of sources to produce
        self.S, self.N, self.L, self.B, self.H, self.P = S, N, L, B, H, P
        self.X, self.R = X, R

        # Front end
        self.fe = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=N,
                      kernel_size=L, stride=L // 2, padding=L // 2),
            nn.ReLU(),
        ])

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobalLayerNorm(N)
        # self.ln = nn.BatchNorm1d(N)
        self.l1 = nn.Conv1d(in_channels=N, out_channels=B, kernel_size=1)

        # Separation module
        self.sm = nn.ModuleList([
            TDCN.TCN(B=B, H=H, P=P, D=2 ** d)
            for _ in range(R) for d in range(X)])

        if B != N:
            # self.ln_bef_out_reshape = GlobalLayerNorm(B)
            self.reshape_before_masks = nn.Conv1d(in_channels=B,
                                                  out_channels=N,
                                                  kernel_size=1)
            # self.ln_bef_masks = nn.GlobalLayerNorm(S * N)

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=S,
                           kernel_size=(N + 1, 1),
                           padding=(N - N // 2, 0))

        # Back end
        self.be = nn.ConvTranspose1d(in_channels=N * S, out_channels=S,
                                     output_padding=(L // 2) - 1, kernel_size=L,
                                     stride=L // 2, padding=L // 2,
                                     groups=S)
        # self.ln_mask_in = nn.BatchNorm1d(self.N)
        self.ln_mask_in = GlobalLayerNorm(self.N)

    # Forward pass
    def forward(self, x):
        # Front end
        for l in self.fe:
            x = l(x)

        # Split paths
        s = x.clone()

        # Separation module
        x = self.ln(x)
        x = self.l1(x)
        for l in self.sm:
            x = l(x)

        if self.B != self.N:
            # x = self.ln_bef_out_reshape(x)
            x = self.reshape_before_masks(x)

        x = self.ln_mask_in(x)
        # Get masks and apply them
        x = self.m(x.unsqueeze(1))
        x = nn.functional.relu(x)
        if self.S == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)
        x = x * s.unsqueeze(1)
        del s

        # Back end
        return self.be(x.view(x.shape[0], -1, x.shape[-1]))

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
        dir_id = 'tasnet_L_{}_N_{}'.format(sample_res, freq_res)
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
        model_dir_name = 'tasnet_L_{}_N_{}'.format(model.L, model.N)
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


class ResidualTN(nn.Module):

    # Simplified TCN layer
    class TCN(nn.Module):
        def __init__(self, B, H, P, D):
            super(ResidualTN.TCN, self).__init__()

            self.m = nn.ModuleList([
                nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1),
                nn.PReLU(),
                # GlobalLayerNorm(H),
                CepstralNorm(H),
                nn.Conv1d(in_channels=H, out_channels=H, kernel_size=P,
                          padding=(D * (P - 1)) // 2, dilation=D, groups=H),
                nn.PReLU(),
                # GlobalLayerNorm(H),
                CepstralNorm(H),
                nn.Conv1d(in_channels=H, out_channels=B, kernel_size=1),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l(y)
            return x + y

    # Set things up
    def __init__(self, N, L, B, H, P, X, R, S=1):
        super(ResidualTN, self).__init__()

        # Number of sources to produce
        self.S, self.N, self.L, self.B, self.H, self.P = S, N, L, B, H, P
        self.X, self.R = X, R

        # Front end
        self.fe = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=N,
                      kernel_size=L, stride=L // 2, padding=L // 2),
            nn.ReLU(),
        ])

        self.ln = nn.BatchNorm1d(N)
        self.l1 = nn.Conv1d(in_channels=N, out_channels=B, kernel_size=1)

        # Separation module
        # Residual connections
        self.residual_to_from = [[] for _ in range(R*X)]
        self.residual_to_from[8] = [-1]
        self.residual_to_from[16] = [-1, 8]
        self.residual_to_from[24] = [-1, 8, 16]
        self.residual_to_from[11] = [3]
        self.residual_to_from[19] = [3, 11]
        self.residual_to_from[27] = [3, 11, 19]
        self.layer_to_dense = {}
        j = 0
        for i, res_connections in enumerate(self.residual_to_from):
            if len(res_connections):
                self.layer_to_dense[i] = j
                j += 1

        self.residual_denses = nn.ModuleList([
            nn.Conv1d(in_channels=len(res_connections) * B,
                      out_channels=B, kernel_size=1)
            for res_connections in self.residual_to_from
            if len(res_connections) > 0
        ])

        self.prev_connections = {}
        self.residual_norms = []
        k = 0
        for res_from in self.residual_to_from:
            for res_ind in res_from:
                if res_ind not in self.prev_connections:
                    self.prev_connections[res_ind] = k
                    k += 1
                    self.residual_norms.append(CepstralNorm(B))
        self.residual_norms = nn.ModuleList(self.residual_norms)

        self.sm = nn.ModuleList(
            [ResidualTN.TCN(B=B, H=H, P=P, D=2 ** d)
             for _ in range(R) for d in range(X)])

        if B != N:
            self.reshape_before_masks = nn.Conv1d(in_channels=B,
                                                  out_channels=N,
                                                  kernel_size=1)

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=S,
                           kernel_size=(N + 1, 1),
                           padding=(N - N // 2, 0))

        # Back end
        self.be = nn.ConvTranspose1d(in_channels=N * S, out_channels=S,
                                     output_padding=(L // 2) - 1, kernel_size=L,
                                     stride=L // 2, padding=L // 2,
                                     groups=S)
        self.ln_mask_in = nn.BatchNorm1d(self.N)

    # Forward pass
    def forward(self, x):
        # Front end
        for l in self.fe:
            x = l(x)

        # Split paths
        encoded_mixture = x.clone()

        # Separation module
        x = self.ln(x)
        x = self.l1(x)
        separation_input = x.clone()

        layer_outputs = []
        for l, tcn in enumerate(self.sm):
            # gather residuals
            residual_outputs = []
            for k, res_ind in enumerate(self.residual_to_from[l]):
                if res_ind == -1:
                    residual_outputs.append(self.residual_norms[
                        self.prev_connections[res_ind]](
                        separation_input))
                else:
                    residual_outputs.append(self.residual_norms[
                        self.prev_connections[res_ind]](
                        layer_outputs[res_ind]))

            if residual_outputs:
                if len(residual_outputs) == 1:
                    residuals = residual_outputs[0]
                else:
                    # Before concatenation normalize everything
                    residuals = torch.cat(residual_outputs, dim=1)
                x = tcn(x + self.residual_denses[
                    self.layer_to_dense[l]](residuals))
            else:
                x = tcn(x)
            if l in [8, 16, 24, 3, 11, 19]:
                layer_outputs.append(x.clone())
            else:
                layer_outputs.append(None)

        if self.B != self.N:
            # x = self.ln_bef_out_reshape(x)
            x = self.reshape_before_masks(x)

        x = self.ln_mask_in(x)
        # Get masks and apply them
        x = self.m(x.unsqueeze(1))
        x = nn.functional.relu(x)
        if self.S == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)
        x = x * encoded_mixture.unsqueeze(1)
        del encoded_mixture

        # Back end
        return self.be(x.view(x.shape[0], -1, x.shape[-1]))

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
        dir_id = 'residualTN_new_L_{}_N_{}'.format(sample_res, freq_res)
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
        model_dir_name = 'residualTN_new_L_{}_N_{}'.format(
            model.L, model.N)
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


if __name__ == "__main__":
    import torch
    import os, sys
    model = TDCN(
        B=256,
        H=512,
        P=3,
        R=4,
        X=8,
        L=21,
        N=256,
        S=2)

    print('Testing Forward pass')
    if sys.argv[1] == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
        model = model.cuda()
        dummy_input = torch.rand(1, 1, 32000).cuda()
    elif sys.argv[1] == 'cpu':
        dummy_input = torch.rand(1, 1, 32000)

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
