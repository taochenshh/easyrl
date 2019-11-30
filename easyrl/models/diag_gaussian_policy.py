import torch
import torch.nn as nn
from easyrl.utils.torch_util import TanhTransform
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import TransformedDistribution


class DiagGaussianPolicy(nn.Module):
    def __init__(self,
                 body_net,
                 action_dim,
                 init_log_std=-0.51,
                 std_cond_in=False,
                 tanh_on_dist=False):  # add tanh on the action distribution
        super().__init__()
        self.std_cond_in = std_cond_in
        self.tanh_on_dist = tanh_on_dist
        self.body = body_net

        feature_dim = None
        for i in reversed(range(len(self.body.fcs))):
            layer = self.body.fcs[i]
            if hasattr(layer, 'out_features'):
                feature_dim = layer.out_features
                break

        self.head_mean = nn.Linear(feature_dim, action_dim)
        if self.std_cond_in:
            self.head_logstd = nn.Linear(feature_dim, action_dim)
        else:
            self.head_logstd = nn.Parameter(torch.full((action_dim,),
                                                       init_log_std))

    def forward(self, x=None, body_x=None):
        if x is None and body_x is None:
            raise ValueError('One of [x, body_x] should be provided!')
        if body_x is None:
            body_x = self.body(x)
        mean = self.head_mean(body_x)
        if self.std_cond_in:
            log_std = self.head_logstd(body_x)
        else:
            log_std = self.head_logstd.expand_as(mean)
        std = torch.exp(log_std)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        if self.tanh_on_dist:
            action_dist = TransformedDistribution(action_dist,
                                                  [TanhTransform(cache_size=1)])
        return action_dist, body_x
