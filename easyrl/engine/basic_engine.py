from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from easyrl.configs import cfg
from easyrl.utils.rl_logger import TensorboardLogger


@dataclass
class BasicEngine:
    agent: Any
    runner: Any

    def __post_init__(self):
        self.cur_step = 0
        self._best_eval_ret = -np.inf
        self._eval_is_best = False
        if cfg.alg.test or cfg.alg.resume:
            self.cur_step = self.agent.load_model(step=cfg.alg.resume_step)
        else:
            if cfg.alg.pretrain_model is not None:
                self.agent.load_model(pretrain_model=cfg.alg.pretrain_model)
            cfg.alg.create_model_log_dir()
        self.train_ep_return = deque(maxlen=100)
        self.smooth_eval_return = None
        self.smooth_tau = cfg.alg.smooth_eval_tau
        self.optim_stime = None
        if not cfg.alg.test:
            self.tf_logger = TensorboardLogger(log_dir=cfg.alg.log_dir)

    def train(self, **kwargs):
        raise NotImplementedError

    def eval(self, **kwargs):
        raise NotImplementedError
