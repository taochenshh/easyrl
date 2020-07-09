import time
from copy import deepcopy

import numpy as np
import torch
from gym.wrappers.time_limit import TimeLimit
from easyrl.runner.base_runner import BasicRunner
from easyrl.utils.data import StepData
from easyrl.utils.data import Trajectory
from easyrl.utils.torch_util import torch_to_np

class StepRunner(BasicRunner):
    # Simulate the environment for T steps,
    # and in the next call, the environment will continue
    # from where it's left in the previous call.
    # only single env (no parallel envs) is supported for now.
    # we also assume the environment is wrapped by TimeLimit
    # from https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
    def __init__(self, agent, env, eval_env=None):
        super().__init__(agent=agent,
                         env=env,
                         eval_env=eval_env)
        self.step_data = None
        if not (isinstance(env, TimeLimit) and isinstance(eval_env, TimeLimit)):
            raise TypeError('Please add TimeLimit wrapper on the environment.')

    @torch.no_grad()
    def __call__(self, time_steps, sample=True, evaluation=False,
                 return_on_done=False, render=False, render_image=False,
                 sleep_time=0, reset_kwargs=None, action_kwargs=None):
        traj = Trajectory()
        if reset_kwargs is None:
            reset_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        if evaluation:
            env = self.eval_env
        else:
            env = self.train_env
        if self.step_data is None or evaluation:
            ob = env.reset(**reset_kwargs)
        else:
            ob = self.step_data.ob
        ob = deepcopy(ob)
        for t in range(time_steps):
            if render:
                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if render_image:
                # get render images at the same time step as ob
                imgs = deepcopy(env.get_images())

            action, action_info = self.agent.get_action(ob,
                                                        sample=sample,
                                                        **action_kwargs)
            next_ob, reward, done, info = env.step(action)
            next_ob = deepcopy(next_ob)
            if render_image:
                for img, inf in zip(imgs, info):
                    inf['render_image'] = deepcopy(img)

            sd = StepData(ob=ob,
                          action=deepcopy(action),
                          action_info=deepcopy(action_info),
                          next_ob=next_ob,
                          reward=deepcopy(reward),
                          done=deepcopy(done) and not info.get('TimeLimit.truncated',
                                                               False),
                          info=deepcopy(info))
            ob = next_ob
            traj.add(sd)
            if return_on_done and done:
                break
            if done:
                ob = deepcopy(env.reset(**reset_kwargs))
        self.step_data = deepcopy(traj[-1])
