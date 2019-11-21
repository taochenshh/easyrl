import torch
import torch.nn as nn
from torch.nn import functional as F
from easyrl.models.mlp import MLP
from torch.distributions import Independent
from torch.distributions import Normal

class DiagGaussianPolicy(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 hidden_act=F.relu,
                 output_act=None,
                 add_layer_norm=False,
                 add_spectral_norm=False,
                 init_log_std=-0.22,
                 std_cond_in=False):
        super().__init__()
        self.std_cond_in = std_cond_in
        self.output_act = output_act
        self.base = MLP(input_size=input_size,
                        hidden_sizes=hidden_sizes[:-1],
                        output_size=hidden_sizes[-1],
                        hidden_act=hidden_act,
                        output_act=hidden_act,
                        add_layer_norm=add_layer_norm,
                        add_spectral_norm=add_spectral_norm)
        self.head_mean = nn.Linear(hidden_sizes[-1], output_size)
        if self.std_cond_in:
            self.head_logstd = nn.Linear(hidden_sizes[-1], output_size)
        else:
            self.head_logstd = nn.Parameter(torch.full((output_size,),
                                                       init_log_std))

    def forward(self, x):
        x = self.base(x)
        mean = self.head_mean(x)
        if self.std_cond_in:
            log_std = self.head_logstd(x)
        else:
            log_std = self.head_logstd.expand_as(mean)
        std = torch.exp(log_std)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        return action_dist