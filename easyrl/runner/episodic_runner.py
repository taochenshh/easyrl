import time
from copy import deepcopy

import numpy as np
import torch

from easyrl.runner.base_runner import BasicRunner
from easyrl.utils.data import StepData
from easyrl.utils.data import Trajectory
from easyrl.utils.gym_util import get_render_images
from easyrl.utils.torch_util import torch_to_np


class EpisodicRunner(BasicRunner):
    @torch.no_grad()
    def __call__(self, time_steps, sample=True, evaluation=False,
                 return_on_done=False, render=False, render_image=False,
                 sleep_time=0, reset_kwargs=None, action_kwargs=None,
                 random_action=False):
        traj = Trajectory()
        if reset_kwargs is None:
            reset_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        if evaluation:
            env = self.eval_env
        else:
            env = self.train_env
        ob = env.reset(**reset_kwargs)
        # this is critical for some environments depending
        # on the returned ob data. use deepcopy() to avoid
        # adding the same ob to the traj

        # only add deepcopy() when a new ob is generated
        # so that traj[t].next_ob is still the same instance as traj[t+1].ob
        ob = deepcopy(ob)
        if return_on_done:
            all_dones = np.zeros(env.num_envs, dtype=bool)
        for t in range(time_steps):
            if render:
                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if render_image:
                # get render images at the same time step as ob
                imgs = get_render_images(env)

            if random_action:
                action = env.random_actions()
                action_info = dict()
            else:
                action, action_info = self.agent.get_action(ob,
                                                            sample=sample,
                                                            **action_kwargs)
            next_ob, reward, done, info = env.step(action)
            next_ob = deepcopy(next_ob)
            if render_image:
                for img, inf in zip(imgs, info):
                    inf['render_image'] = deepcopy(img)

            done_idx = np.argwhere(done).flatten()
            if done_idx.size > 0 and return_on_done:
                # vec env automatically resets the environment when it's done
                # so the returned next_ob is not actually the next observation
                all_dones[done_idx] = True

            true_done = deepcopy(done)
            for iidx, inf in enumerate(info):
                true_done[iidx] = true_done[iidx] and not inf.get('TimeLimit.truncated',
                                                                  False)
            sd = StepData(ob=ob,
                          action=deepcopy(action),
                          action_info=deepcopy(action_info),
                          next_ob=next_ob,
                          reward=deepcopy(reward),
                          done=true_done,
                          info=deepcopy(info))
            ob = next_ob
            traj.add(sd)
            if return_on_done and np.all(all_dones):
                break
        if not evaluation:
            last_val = self.agent.get_val(traj[-1].next_ob)
            traj.add_extra('last_val', torch_to_np(last_val))
        return traj

    def reset(self, *args, **kwargs):
        pass
