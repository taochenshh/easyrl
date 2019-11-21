import torch.nn as nn
from torch.distributions import Categorical


class CategoricalPolicy(nn.Module):
    def __init__(self,
                 body_net,
                 action_dim):  # add tanh on the action distribution
        super().__init__()
        self.body = body_net
        feature_dim = None
        for i in reversed(range(len(self.body.fcs))):
            layer = self.body.fcs[i]
            if hasattr(layer, 'out_features'):
                feature_dim = layer.out_features
                break
        self.head = nn.Linear(feature_dim, action_dim)

    def forward(self, x=None, body_x=None):
        if x is None and body_x is None:
            raise ValueError('One of [x, body_x] should be provided!')
        if body_x is None:
            body_x = self.body(x)
        pi = self.head(body_x)
        action_dist = Categorical(logits=pi)
        return action_dist, body_x
