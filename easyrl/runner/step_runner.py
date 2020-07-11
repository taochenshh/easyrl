import time
from copy import deepcopy

import torch
from gym.wrappers.time_limit import TimeLimit

from easyrl.runner.base_runner import BasicRunner
from easyrl.utils.common import list_to_numpy
from easyrl.utils.data import StepData
from easyrl.utils.data import Trajectory


class StepRunner(BasicRunner):
    # Simulate the environment for T steps,
    # and in the next call, the environment will continue
    # from where it's left in the previous call.
    # only single env (no parallel envs) is supported for now.
    # we also assume the environment is wrapped by TimeLimit
    # from https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
    def __init__(self, agent, env, eval_env=None, max_steps=None):
        super().__init__(agent=agent,
                         env=env,
                         eval_env=eval_env)
        self.cur_ob = None
        self.max_steps = max_steps
        self.cur_step = 0
        if not (isinstance(env, TimeLimit) and isinstance(eval_env, TimeLimit)):
            raise TypeError('Please add TimeLimit wrapper on the environment.')

    @torch.no_grad()
    def __call__(self, time_steps, sample=True, evaluation=False,
                 return_on_done=False, render=False, render_image=False,
                 sleep_time=0, reset_kwargs=None,
                 action_kwargs=None, random_action=False):
        traj = Trajectory()
        if reset_kwargs is None:
            reset_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        if evaluation:
            env = self.eval_env
        else:
            env = self.train_env
        if self.cur_ob is None or evaluation:
            ob = env.reset(**reset_kwargs)
            self.cur_step = 0
        else:
            ob = self.cur_ob
        ob = deepcopy(ob)
        for t in range(time_steps):
            if render:
                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if render_image:
                # get render images at the same time step as ob
                imgs = deepcopy(env.get_images())
            if random_action:
                action = env.action_space.sample()
                action_info = dict()
            else:
                action, action_info = self.agent.get_action(ob,
                                                            sample=sample,
                                                            **action_kwargs)
            next_ob, reward, done, info = env.step(action)
            self.cur_step += 1
            next_ob = deepcopy(next_ob)
            if render_image:
                for img, inf in zip(imgs, info):
                    inf['render_image'] = deepcopy(img)
            true_done = done and not info.get('TimeLimit.truncated',
                                              False)
            sd = StepData(ob=list_to_numpy(deepcopy(ob),
                                           expand_dims=0),
                          action=list_to_numpy(deepcopy(action),
                                               expand_dims=0),
                          action_info=[deepcopy(action_info)],
                          next_ob=list_to_numpy(deepcopy(next_ob),
                                                expand_dims=0),
                          reward=list_to_numpy(reward),
                          done=list_to_numpy(true_done),
                          info=[deepcopy(info)])
            ob = next_ob
            traj.add(sd)
            if return_on_done and done:
                break
            need_reset = done
            if self.max_steps is not None:
                need_reset = need_reset or self.cur_step > self.max_steps
            if need_reset:
                ob = deepcopy(env.reset(**reset_kwargs))
                self.cur_step = 0
        self.cur_ob = deepcopy(ob)
        return traj

    def reset(self, reset_kwargs=None):
        if reset_kwargs is None:
            reset_kwargs = {}
        ob = self.train_env.reset(**reset_kwargs)
        self.cur_step = 0
        self.cur_ob = deepcopy(ob)
