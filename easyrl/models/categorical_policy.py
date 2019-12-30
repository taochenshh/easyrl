import torch.nn as nn
from torch.distributions import Categorical


class CategoricalPolicy(nn.Module):
    def __init__(self,
                 body_net,
                 action_dim,
                 in_features=None,
                 ):  # add tanh on the action distribution
        super().__init__()
        self.body = body_net
        if in_features is None:
            for i in reversed(range(len(self.body.fcs))):
                layer = self.body.fcs[i]
                if hasattr(layer, 'out_features'):
                    in_features = layer.out_features
                    break

        self.head = nn.Linear(in_features, action_dim)

    def forward(self, x=None, body_x=None, **kwargs):
        if x is None and body_x is None:
            raise ValueError('One of [x, body_x] should be provided!')
        if body_x is None:
            body_x = self.body(x, **kwargs)
        pi = self.head(body_x)
        action_dist = Categorical(logits=pi)
        return action_dist, body_x
