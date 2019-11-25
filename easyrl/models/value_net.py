import torch.nn as nn


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

    def forward(self, x):
        x = self.body(x)
        val = self.head(x)
        return val
