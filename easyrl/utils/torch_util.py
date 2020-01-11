import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Independent
from torch.distributions import Transform
from torch.distributions import TransformedDistribution
from torch.distributions import constraints
from torch.nn.functional import softplus
from torch.utils.data import Dataset

from easyrl.utils.rl_logger import logger


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def load_torch_model(model_file):
    logger.info(f'Loading model from {model_file}')
    if isinstance(model_file, str):
        model_file = Path(model_file)
    if not model_file.exists():
        raise ValueError(f'Checkpoint file ({model_file}) '
                         f'does not exist!')
    ckpt_data = torch.load(model_file)
    return ckpt_data


def torch_to_np(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('tensor has to be a torch tensor!')
    return tensor.cpu().detach().numpy()


def action_from_dist(action_dist, sample=True):
    if isinstance(action_dist, Categorical):
        if sample:
            return action_dist.sample()
        else:
            return action_dist.probs.argmax(dim=-1)
    elif isinstance(action_dist, Independent):
        if sample:
            return action_dist.rsample()
        else:
            return action_dist.mean
    elif isinstance(action_dist, TransformedDistribution):
        if not sample:
            raise TypeError('Deterministic sampling is not '
                            'defined for transformed distribution!')
        if action_dist.has_rsample:
            return action_dist.rsample()
        else:
            return action_dist.sample()
    else:
        raise TypeError('Getting actions for the given '
                        'distribution is not implemented!')


def action_log_prob(action, action_dist):
    if isinstance(action_dist, Categorical):
        log_prob = action_dist.log_prob(action)
        return log_prob
    elif isinstance(action_dist,
                    Independent) or isinstance(action_dist,
                                               TransformedDistribution):
        log_prob = action_dist.log_prob(action)
        return log_prob
    else:
        raise TypeError('Getting log_prob of actions for the given '
                        'distribution is not implemented!')


def action_entropy(action_dist, log_prob=None):
    try:
        entropy = action_dist.entropy()
    except NotImplementedError:
        # Some transformations might not have entropy implemented
        # such as the tanh normal distribution (transformed distribution)
        # TODO which one to use (zero or one sample)
        entropy = torch.zeros(tuple(action_dist.batch_shape))
        # try using one sample to approximate the entropy (monte carlo)
        # entropy = -log_prob
    return entropy


class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        eps = torch.finfo(y.dtype).eps
        return self.atanh(y.clamp(min=-1. + eps, max=1. - eps))

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable,
        # see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - softplus(-2. * x))


def cosine_similarity(x1, x2):
    """

    Args:
        x1: shape [M, N]
        x2: shape [K, N]

    Returns:
        shape [M, K]
    """
    x1 = F.normalize(x1, p=2, dim=1)
    x2 = F.normalize(x2, p=2, dim=1)
    cos_sim = torch.mm(x1, x2.transpose(0, 1))
    return cos_sim


def cdist_l2(x1, x2):
    """
    pairwise l2 distance between x1 and x2

    Args:
        x1: shape [M, N]
        x2: shape [K, N]

    Returns:
        shape [M, K]
    """
    # on pytorch 1.3 and cuda 10
    # this function is much faster then torch.cdist
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res


def batched_cdist_l2(x1, x2):
    """
    cdist in batch mode
    [
     cdist(x1[0], x2[0]),
     cdist(x1[1], x2[1]),
     cdist(x1[2], x2[2]),
     ...
    ]

    Args:
        x1: shape [B, M, N]
        x2: shape [B, K, N]

    Returns:
        shape [B, M, K]
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res


def ortho_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
    r"""Applies orthogonal initialization for the parameters of a given module.

    Args:
        module (nn.Module): A module to apply orthogonal initialization over its parameters.
        nonlinearity (str, optional): Nonlinearity followed by forward pass of the module. When nonlinearity
            is not ``None``, the gain will be calculated and :attr:`weight_scale` will be ignored.
            Default: ``None``
        weight_scale (float, optional): Scaling factor to initialize the weight. Ignored when
            :attr:`nonlinearity` is not ``None``. Default: 1.0
        constant_bias (float, optional): Constant value to initialize the bias. Default: 0.0

    .. note::

        Currently, the only supported :attr:`module` are elementary neural network layers, e.g.
        nn.Linear, nn.Conv2d, nn.LSTM. The submodules are not supported.

    Example::

        >>> a = nn.Linear(2, 3)
        >>> ortho_init(a)

    """
    if nonlinearity is not None:
        gain = nn.init.calculate_gain(nonlinearity)
    else:
        gain = weight_scale

    if isinstance(module, (nn.RNNBase, nn.RNNCellBase)):
        for name, param in module.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias_' in name:
                nn.init.constant_(param, constant_bias)
    else:  # other modules with single .weight and .bias
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, constant_bias)


class EpisodeDataset(Dataset):
    def __init__(self, **kwargs):
        self.data = dict()
        for key, val in kwargs.items():
            self.data[key] = self._swap_leading_axes(val)
        self.length = next(iter(self.data.values())).shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = dict()
        for key, val in self.data.items():
            sample[key] = val[idx]
        return sample

    def _swap_leading_axes(self, array):
        """
        Swap and then flatten the array along axes 0 and 1

        Args:
            array (np.ndarray): array data

        Returns:
            np.ndarray: reshaped array
        """
        s = array.shape
        return array.swapaxes(0, 1).reshape(s[0] * s[1],
                                            *s[2:])
