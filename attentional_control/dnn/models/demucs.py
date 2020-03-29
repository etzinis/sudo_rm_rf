"""!
@brief Demucs code copied from:
https://github.com/facebookresearch/demucs

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
import functools
import glob2
import os, sys
import datetime


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
    weight = torch.arange(stride, device=x.device, dtype=torch.float) / stride
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
                 sources=2,
                 audio_channels=1,
                 channels=80,
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
            x = torch.cat([x, skip], dim=1)
            # x = th.cat([x, saved.pop(-1)], dim=1)
            x = self.final(x)
            # print('after final')
            # print(x.shape)

        x = x.view(x.size(0), self.sources, x.size(-1))
        return center_trim(x, mix)

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
        model = cls()
        model.load_state_dict(package['state_dict'])
        return model

    @classmethod
    def load_best_model(cls, models_dir):
        dir_id = 'demucs'
        dir_path = os.path.join(models_dir, dir_id)
        best_path = glob2.glob(dir_path + '/best_*')[0]
        return cls.load(best_path)

    @classmethod
    def load_latest_model(cls, models_dir):
        dir_id = 'demucs'
        dir_path = os.path.join(models_dir, dir_id)
        latest_path = glob2.glob(dir_path + '/current_*')[0]
        return cls.load(latest_path)

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
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
    def encode_dir_name(cls):
        model_dir_name = 'demucs'
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
                     tr_loss, cv_loss, cv_loss_name,
                     cometml_experiment=None, model_name='demucs'):

        model_dir_path = os.path.join(save_dir, cls.encode_dir_name())
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
            best_model_id = 'best_' + file_id + '.pt'
            save_path = os.path.join(model_dir_path, best_model_id)
            cls.save(model, save_path, optimizer, epoch,
                     tr_loss=tr_loss, cv_loss=cv_loss)
            print('===> Saved best model at: {}\n'.format(save_path))

            if cometml_experiment is not None:
                print('Trying to upload best model to cometml...')
                cometml_experiment.log_model(model_name,
                                             save_path,
                                             file_name='best_model',
                                             overwrite=True,
                                             metadata=None, copy_to_tmp=True)

        current_model_id = 'current_' + file_id + '.pt'
        save_path = os.path.join(model_dir_path, current_model_id)
        cls.save(model, save_path, optimizer, epoch,
                 tr_loss=tr_loss, cv_loss=cv_loss)
        if cometml_experiment is not None:
            print('Trying to upload current model to cometml...')
            cometml_experiment.log_model(model_name,
                                         save_path,
                                         file_name=current_model_id,
                                         overwrite=True,
                                         metadata=None, copy_to_tmp=True)


if __name__ == "__main__":
    import torch
    import os
    model = Demucs(sources=2,
                     audio_channels=1,
                     channels=80,
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
    print('Testing Forward pass')
    if sys.argv[1] == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
        model = model.cuda()
        dummy_input = torch.rand(1, 32000).cuda()
    elif sys.argv[1] == 'cpu':
        dummy_input = torch.rand(1, 32000)

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
