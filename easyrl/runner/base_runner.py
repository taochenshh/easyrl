import numpy as np
from copy import deepcopy
from easyrl.utils.gym_util import get_true_done
from collections import deque

class BasicRunner:
    def __init__(self, agent, env, eval_env=None):
        self.agent = agent
        self.train_env = env
        self.num_train_envs = env.num_envs
        self.obs = None
        self.eval_env = env if eval_env is None else eval_env
        self.train_ep_return = deque(maxlen=100)
        self.train_ep_len = deque(maxlen=100)
        self.reset_record()

    def __call__(self, **kwargs):
        raise NotImplementedError

    def reset(self, env=None, *args, **kwargs):
        if env is None:
            env = self.train_env
        self.obs = env.reset(*args, **kwargs)
        self.reset_record()

    def reset_record(self):
        self.cur_ep_len = np.zeros(self.num_train_envs)
        self.cur_ep_return = np.zeros(self.num_train_envs)

    def get_true_done_next_ob(self, next_ob, done, reward, info, all_dones):
        done_idx = np.argwhere(done).flatten()
        self.cur_ep_len += 1
        if 'raw_reward' in info[0]:
            self.cur_ep_return += np.array([x['raw_reward'] for x in info])
        else:
            self.cur_ep_return += reward
        if done_idx.size > 0:
            # vec env automatically resets the environment when it's done
            # so the returned next_ob is not actually the next observation
            true_next_ob = deepcopy(next_ob)
            true_next_ob[done_idx] = np.array([info[i]['true_next_ob'] for i in done_idx])
            if all_dones is not None:
                all_dones[done_idx] = True
            for dix in done_idx:
                self.train_ep_return.append(self.cur_ep_return[dix])
                self.train_ep_len.append(self.cur_ep_len[dix])
            self.cur_ep_return[done_idx] = 0
            self.cur_ep_len[done_idx] = 0
            true_done = deepcopy(done)
            for iidx, inf in enumerate(info):
                true_done[iidx] = get_true_done(true_done[iidx], inf)
        else:
            true_next_ob = next_ob
            true_done = done
        return true_next_ob, true_done, all_dones
