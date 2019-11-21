import torch.nn as nn
from torch.nn import functional as F
from easyrl.utils.rl_logger import logger
from torch.nn.utils.spectral_norm import spectral_norm


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 hidden_act=F.relu,
                 output_act=None,
                 add_layer_norm=False,
                 add_spectral_norm=False):
        super().__init__()
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.add_layer_norm = add_layer_norm
        if add_spectral_norm:
            logger.info('Spectral Normalization on!')
        if add_layer_norm:
            logger.info('Layer Normalization on!')
        in_size = input_size
        self.fcs = []
        self.layer_norms = []
        for i, hid_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, hid_size)
            if add_spectral_norm:
                fc = spectral_norm(fc)
            in_size = hid_size
            self.__setattr__('fc{}'.format(i), fc)
            self.fcs.append(fc)
            if self.add_layer_norm:
                ln = nn.LayerNorm(hid_size)
                self.__setattr__('layer_norm{}'.format(i), ln)
                self.layer_norms.append(ln)
        self.last_fc = nn.Linear(in_size, output_size)
        if add_spectral_norm:
            self.last_fc = spectral_norm(self.last_fc)

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if self.require_layer_norm:
                x = self.layer_norms[i](x)
            x = self.hidden_act(x)
        out = self.last_fc(x)
        if self.output_act is not None:
            out = self.output_act(out)
        return out