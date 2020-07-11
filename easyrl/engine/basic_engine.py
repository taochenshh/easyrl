import numpy as np


class BasicEngine:
    def __init__(self, agent, runner, **kwargs):
        self.agent = agent
        self.runner = runner
        self.cur_step = 0
        self._best_eval_ret = -np.inf
        self._eval_is_best = False

    def train(self, **kwargs):
        raise NotImplementedError

    def eval(self, **kwargs):
        raise NotImplementedError
