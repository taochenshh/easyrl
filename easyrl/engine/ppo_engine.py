import time

from easyrl.configs.ppo_config import ppo_cfg
from easyrl.engine.basic_engine import BasicEngine
from easyrl.utils.gae import cal_gae
from easyrl.utils.torch_util import torch_to_np


class PPOEngine(BasicEngine):
    def __init__(self, agent, env, runner):
        super().__init__(agent=agent,
                         env=env,
                         runner=runner)

    def train(self, **kwargs):
        t0 = time.perf_counter()

        traj = self.runner(ppo_cfg.episode_steps)
        rewards = traj.rewards
        actions_info = traj.actions_info
        vals = np.array([ainfo['val'] for ainfo in actions_info])
        log_prob = np.array([ainfo['log_prob'] for ainfo in actions_info])
        act_dist, last_val = self.agent.get_act_val(traj[-1].next_ob)
        adv = cal_gae(gamma=ppo_cfg.rew_discount,
                      lam=ppo_cfg.gae_lambda,
                      rewards=rewards,
                      value_estimates=vals,
                      last_value=torch_to_np(last_val),
                      dones=traj.dones)
        ret = adv + vals
        data = dict(
            ob=traj.obs,
            action=traj.actions,
            ret=ret,
            adv=adv,
            log_prob=log_prob,
            val=val
        )
        optim_info = self.agent.optimize(data)
        t1 = time.perf_counter()
        return optim_info


