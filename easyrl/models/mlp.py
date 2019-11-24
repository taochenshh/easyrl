import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

from easyrl.utils.rl_logger import logger
from easyrl.utils.torch_util import ortho_init

class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 hidden_act=nn.ReLU,
                 output_act=None,
                 add_layer_norm=False,
                 add_spectral_norm=False):
        super().__init__()
        if not isinstance(hidden_sizes, list):
            raise TypeError('hidden_sizes should be a list')
        if add_spectral_norm:
            logger.info('Spectral Normalization on!')
        if add_layer_norm:
            logger.info('Layer Normalization on!')
        in_size = input_size
        self.fcs = nn.ModuleList()
        for i, hid_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, hid_size)
            # TODO make it nicer
            ortho_init(fc, nonlinearity='tanh', constant_bias=0.0)
            if add_spectral_norm:
                fc = spectral_norm(fc)
            in_size = hid_size
            self.fcs.append(fc)
            if add_layer_norm:
                self.fcs.append(nn.LayerNorm(hid_size))
            self.fcs.append(hidden_act())

        last_fc = nn.Linear(in_size, output_size)
        if add_spectral_norm:
            last_fc = spectral_norm(last_fc)
        self.fcs.append(last_fc)
        if output_act is not None:
            self.fcs.append(output_act())

    def forward(self, x):
        for i, layer in enumerate(self.fcs):
            x = layer(x)
        return x
