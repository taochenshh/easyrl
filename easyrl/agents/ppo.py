import torch

from easyrl.agents.base_agent import BaseAgent
from easyrl.configs.ppo_config import ppo_cfg
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob


class PPOAgent(BaseAgent):
    def __init__(self, actor, critic, same_body=False):
        self.actor = actor
        self.critic = critic
        self.same_body = same_body

    def get_action(self, ob, sample=True, **kwargs):
        t_ob = torch.from_numpy(ob).float().to(ppo_cfg.device)
        act_dist, body_out = self.actor(t_ob)
        action = action_from_dist(act_dist,
                                  sample=sample)
        log_prob = action_log_prob(action, act_dist)
        entropy = act_dist.entropy()
        if self.same_body:
            val = self.critic(body_x=body_out)
        else:
            val = self.critic(x=t_ob)
        action_info = dict(
            log_prob=log_prob,
            entropy=entropy,
            val=val
        )
        return action, action_info

    def optimize(self, data, **kwargs):

        pass
