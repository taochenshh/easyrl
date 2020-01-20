from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from easyrl.agents.base_agent import BaseAgent
from easyrl.configs.ppo_config import ppo_cfg
from easyrl.utils.common import linear_decay_percent
from easyrl.utils.rl_logger import logger
from easyrl.utils.torch_util import action_entropy
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import load_torch_model
from easyrl.utils.torch_util import torch_float
from easyrl.utils.torch_util import torch_to_np


class SACAgent(BaseAgent):
    def __init__(self, actor, critic, same_body=False):
        pass