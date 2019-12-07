import time
from copy import deepcopy

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
    def __call__(self, time_steps, sample=True,
                 return_on_done=False, render=False, render_image=False,
                 sleep_time=0, **kwargs):
        traj = Trajectory()
        ob = self.env.reset()
        # this is critical for some environments depending
        # on the returned ob data. use deepcopy() to avoid
        # adding the same ob to the traj

        # only add deepcopy() when a new ob is generated
        # so that traj[t].next_ob is still the same instance as traj[t+1].ob
        ob = deepcopy(ob)
        if return_on_done:
            all_dones = np.zeros(self.env.num_envs, dtype=bool)
        for t in range(time_steps):
            if render:
                self.env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if render_image:
                # get render images at the same time step as ob
                imgs = deepcopy(self.env.get_images())

            action, action_info = self.agent.get_action(ob,
                                                        sample=sample)
            next_ob, reward, done, info = self.env.step(action)
            next_ob = deepcopy(next_ob)

            if render_image:
                for img, inf in zip(imgs, info):
                    inf['render_image'] = deepcopy(img)

            done_idx = np.argwhere(done).flatten()
            if done_idx.size > 0 and return_on_done:
                # vec env automatically resets the environment when it's done
                # so the returned next_ob is not actually the next observation
                all_dones[done_idx] = True
            sd = StepData(ob=ob,
                          action=deepcopy(action),
                          action_info=deepcopy(action_info),
                          next_ob=next_ob,
                          reward=deepcopy(reward),
                          done=deepcopy(done),
                          info=deepcopy(info))
            ob = next_ob
            traj.add(sd)
            if return_on_done and np.all(all_dones):
                break
        return traj
