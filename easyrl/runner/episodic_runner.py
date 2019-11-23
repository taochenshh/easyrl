import numpy as np
import torch

from easyrl.runner.base_runner import BasicRunner
from easyrl.utils.data import StepData
from easyrl.utils.data import Trajectory


class EpisodicRunner(BasicRunner):
    def __init__(self, agent, env):
        super().__init__(agent=agent,
                         env=env)

    @torch.no_grad()
    def __call__(self, time_steps, sample=True, return_on_done=False, **kwargs):
        traj = Trajectory()
        ob = self.env.reset()
        if return_on_done:
            all_dones = np.zeros(self.env.num_envs, dtype=bool)
        for t in range(time_steps):
            action, action_info = self.agent.get_action(ob,
                                                        sample=sample)
            next_ob, reward, done, info = self.env.step(action)

            done_idx = np.argwhere(done).flatten()
            if done_idx.size > 0 and return_on_done:
                # vec env automatically resets the environment when it's done
                # so the returned next_ob is not actually the next observation
                all_dones[done_idx] = True
            sd = StepData(ob=ob,
                          action=action,
                          action_info=action_info,
                          next_ob=next_ob,
                          reward=reward,
                          done=done,
                          info=info)
            ob = next_ob
            traj.add(sd)
            if return_on_done and np.all(all_dones):
                break
        return traj
