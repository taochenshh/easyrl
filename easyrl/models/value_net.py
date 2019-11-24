import torch.nn as nn
from easyrl.utils.torch_util import ortho_init


class ValueNet(nn.Module):
    def __init__(self,
                 body_net):  # add tanh on the action distribution
        super().__init__()
        self.body = body_net
        feature_dim = None
        for i in reversed(range(len(self.body.fcs))):
            layer = self.body.fcs[i]
            if hasattr(layer, 'out_features'):
                feature_dim = layer.out_features
                break
        self.head = nn.Linear(feature_dim, 1)
        ortho_init(self.head, weight_scale=1.0, constant_bias=0.0)

    def forward(self, x):
        x = self.body(x)
        val = self.head(x)
        return val
