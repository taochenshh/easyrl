from easyrl.agents.base_agent import BaseAgent
from easyrl.configs.sac_config import sac_cfg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy
from easyrl.utils.torch_util import freeze_model
from easyrl.utils.torch_util import move_to
import itertools
from easyrl.utils.torch_util import torch_float
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import torch_to_np

class SACAgent(BaseAgent):
    def __init__(self, actor, q1, q2):
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.q1_tgt = deepcopy(self.q1)
        self.q2_tgt = deepcopy(self.q2)
        freeze_model(self.q1_tgt)
        freeze_model(self.q2_tgt)
        self.q1_tgt.eval()
        self.q2_tgt.eval()

        move_to([self.actor, self.q1, self.q2, self.q1_tgt, self.q2_tgt],
                device=sac_cfg.device)

        optim_args = dict(
            lr=sac_cfg.actor_lr,
            weight_decay=sac_cfg.weight_decay,
            amsgrad=sac_cfg.use_amsgrad
        )

        self.p_optimizer = optim.Adam(self.actor.parameters(),
                                      **optim_args)
        q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        optim_args['lr'] = sac_cfg.critic_lr
        self.q_optimizer = optim.Adam(q_params, **optim_args)

    @torch.no_grad()
    def get_action(self, ob, sample=True, *args, **kwargs):
        self.eval_mode()
        ob = torch_float(ob, device=sac_cfg.device)
        act_dist = self.actor(ob)[0]
        action = action_from_dist(act_dist,
                                  sample=sample)
        action_info = dict()
        return torch_to_np(action), action_info

    @torch.no_grad()
    def get_val(self, ob, action, tgt=False, q1=True, *args, **kwargs):
        self.eval_mode()
        ob = torch_float(ob, device=sac_cfg.device)
        action = torch_float(action, device=sac_cfg.device)
        idx = 1 if q1 else 2
        tgt_suffix = '_tgt' if tgt else ''
        q_func = getattr(self, f'q{idx}{tgt_suffix}')
        val = q_func((ob, action))[0]
        val = val.squeeze(-1)
        return val



    def train_mode(self):
        self.actor.train()
        self.q1.train()
        self.q2.train()

    def eval_mode(self):
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()

