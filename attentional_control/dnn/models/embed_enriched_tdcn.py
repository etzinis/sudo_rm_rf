"""!
@brief Embeddings enriched multi-scale tasnet guding separation using
label information.

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
            ])

            self.next_layer_dense = nn.Conv1d(in_channels=H,
                                              out_channels=B,
                                              kernel_size=1)

            # self.final_output_dense = nn.Conv1d(in_channels=H,
            #                                     out_channels=B,
            #                                     kernel_size=1)

        def forward(self, x):
            y = x.clone()
            for layer in self.m:
                y = layer(y)

            return self.next_layer_dense(y) + x

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
            self.reshape_before_masks = nn.Conv1d(in_channels=B * X * R,
                                                  out_channels=N,
                                                  kernel_size=1)
            # self.ln_bef_masks = nn.GlobalLayerNorm(S * N)

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=S,
                           kernel_size=(N + 1, 1),
                           padding=(N - N // 2, 0))
        # self.ln_mask_in = nn.BatchNorm1d(self.N)
        self.ln_mask_in = GlobalLayerNorm(self.N)

        # Back end
        self.be = nn.ConvTranspose1d(in_channels=N * S, out_channels=S,
                                     output_padding=(L // 2) - 1, kernel_size=L,
                                     stride=L // 2, padding=L // 2,
                                     groups=S)
        # self.attn = Attention((64000 // (self.L-1)))

        self.embedding_network = nn.ModuleList([
            nn.LeakyReLU(),
            # nn.LayerNorm([X * R, B, (64000 // (self.L-1))]),
            # nn.LayerNorm([X * R, B, 1]),
            nn.Conv2d(in_channels=X * R,
                      out_channels=1,
                      kernel_size=1),
            nn.LeakyReLU(),
            # nn.LayerNorm([1, B, (64000 // (self.L - 1))]),
            nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(B, 1)),
            nn.LeakyReLU(),
            nn.LayerNorm([1, 1, (64000 // (self.L - 1))]),
        ])

        self.logits_layer = nn.Linear((64000 // (self.L-1)), 50)

        # # Masks layer
        # self.m2 = nn.Conv2d(in_channels=1,
        #                     out_channels=S,
        #                     kernel_size=(N + 1, 1),
        #                     padding=(N - N // 2, 0))
        # self.ln_mask_in2 = GlobalLayerNorm(self.N)

    def get_source_embedding(self, source_feature_map):
        x = source_feature_map
        for module in self.embedding_network:
            x = module(x)
        return x

    # Forward pass
    def forward(self, x, return_logits=False):
        # Front end
        for l in self.fe:
            x = l(x)

        # Split paths
        s = x.clone()

        # Separation module
        x = self.ln(x)
        x = self.l1(x)

        accum_outs = []

        # do the forward path and also accumulate all scales representations
        for block in self.sm:
            interm_out = block(x)
            accum_outs.append(interm_out.clone())
            x = interm_out

        multi_scale_maps = torch.stack(accum_outs, dim=1)

        # Extract the masks from the multi-scale maps activations
        x = self.ln_mask_in(x)
        masks_vec = self.m(x.unsqueeze(1))
        if self.S == 1:
            final_masks = torch.sigmoid(x)
        else:
            final_masks = nn.functional.softmax(masks_vec, dim=1)
        x = final_masks * s.unsqueeze(1)
        del s

        if return_logits:
            # apply masks to all multi-scale maps
            sources_logits = []
            for i in range(self.S):
                source_embedding = self.get_source_embedding(
                    multi_scale_maps * final_masks[:, i].unsqueeze(1))
                sources_logits.append(
                    self.logits_layer(source_embedding.view(
                        source_embedding.shape[0], -1)))
            logits = torch.stack(sources_logits, dim=1)

            # # Just use the full multi-scale features as logits
            # sources_embeddings = self.get_source_embedding(
            #     multi_scale_maps)
            # logits = self.logits_layer(sources_embeddings.view(
            #     sources_embeddings.shape[0], -1))

            return self.be(x.view(x.shape[0], -1, x.shape[-1])), logits
        else:
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
        dir_id = 'emb_enriched_tdcn_L_{}_N_{}'.format(sample_res,
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
        model_dir_name = 'emb_enriched_tdcn_L_{}_N_{}'.format(model.L, model.N)
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
    model = TDCN(
        B=256,
        P=3,
        H=512,
        R=3,
        X=8,
        S=2,
        L=21,
        N=256)
    print(model)

    print('Testing Forward pass')
    dummy_input = torch.rand(4, 1, 32000)
    pred_sources, sources_embeddings = model.forward(
        dummy_input, return_logits=True)
    print(pred_sources.size())
    print(sources_embeddings.size())

    numparams = 0
    for f in model.parameters():
        if f.requires_grad:
            numparams += f.numel()
    print('Trainable Parameters: {}'.format(numparams))
