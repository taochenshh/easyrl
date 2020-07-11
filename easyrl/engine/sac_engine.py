import time
from collections import deque
from copy import deepcopy
from itertools import count

import numpy as np
import torch
from tqdm import tqdm

from easyrl.configs.sac_config import sac_cfg
from easyrl.engine.basic_engine import BasicEngine
from easyrl.utils.common import get_list_stats
from easyrl.utils.common import save_traj
from easyrl.utils.data import Trajectory
from easyrl.utils.rl_logger import TensorboardLogger


class SACEngine(BasicEngine):
    def __init__(self, agent, runner, memory):
        super().__init__(agent=agent,
                         runner=runner)
        self.memory = memory
        if sac_cfg.test or sac_cfg.resume:
            self.cur_step = self.agent.load_model(step=sac_cfg.resume_step)
        else:
            if sac_cfg.pretrain_model is not None:
                self.agent.load_model(pretrain_model=sac_cfg.pretrain_model)
            sac_cfg.create_model_log_dir()
        self.train_ep_return = deque(maxlen=100)
        self.smooth_eval_return = None
        self.smooth_tau = sac_cfg.smooth_eval_tau
        self.optim_stime = None
        if not sac_cfg.test:
            self.tf_logger = TensorboardLogger(log_dir=sac_cfg.log_dir)

    def train(self):
        if len(self.memory) < sac_cfg.warmup_steps:
            self.runner.reset()
            traj, _ = self.rollout_once(random_action=True,
                                        time_steps=sac_cfg.warmup_steps - len(self.memory))
            self.add_traj_to_memory(traj)
        self.runner.reset()
        for iter_t in count():
            traj, rollout_time = self.rollout_once(sample=True,
                                                   time_steps=sac_cfg.opt_interval)
            self.add_traj_to_memory(traj)
            train_log_info = self.train_once()
            if iter_t % sac_cfg.eval_interval == 0:
                det_log_info, _ = self.eval(eval_num=sac_cfg.test_num, sample=False)
                sto_log_info, _ = self.eval(eval_num=sac_cfg.test_num, sample=True)
                det_log_info = {f'det/{k}': v for k, v in det_log_info.items()}
                sto_log_info = {f'sto/{k}': v for k, v in sto_log_info.items()}
                eval_log_info = {**det_log_info, **sto_log_info}
                self.agent.save_model(is_best=self._eval_is_best,
                                      step=self.cur_step)
            else:
                eval_log_info = None
            if iter_t % sac_cfg.log_interval == 0:
                train_log_info['train/rollout_time'] = rollout_time
                if eval_log_info is not None:
                    train_log_info.update(eval_log_info)
                scalar_log = {'scalar': train_log_info}
                self.tf_logger.save_dict(scalar_log, step=self.cur_step)
            if self.cur_step > sac_cfg.max_steps:
                break

    @torch.no_grad()
    def eval(self, render=False, save_eval_traj=False, sample=True,
             eval_num=1, sleep_time=0, smooth=True, no_tqdm=None):
        time_steps = []
        rets = []
        lst_step_infos = []
        if no_tqdm:
            disable_tqdm = bool(no_tqdm)
        else:
            disable_tqdm = not sac_cfg.test
        for idx in tqdm(range(eval_num), disable=disable_tqdm):
            traj, _ = self.rollout_once(time_steps=sac_cfg.episode_steps,
                                        return_on_done=True,
                                        sample=sac_cfg.sample_action and sample,
                                        render=render,
                                        sleep_time=sleep_time,
                                        render_image=save_eval_traj,
                                        evaluation=True)
            tsps = traj.steps_til_done.copy().tolist()
            rewards = traj.raw_rewards
            infos = traj.infos
            for ej in range(traj.num_envs):
                ret = np.sum(rewards[:tsps[ej], ej])
                rets.append(ret)
                lst_step_infos.append(infos[tsps[ej] - 1][ej])
            time_steps.extend(tsps)
            if save_eval_traj:
                save_traj(traj, sac_cfg.eval_dir)

        raw_traj_info = {'return': rets,
                         'episode_length': time_steps,
                         'lst_step_info': lst_step_infos}
        log_info = dict()
        for key, val in raw_traj_info.items():
            if 'info' in key:
                continue
            val_stats = get_list_stats(val)
            for sk, sv in val_stats.items():
                log_info['eval/' + key + '/' + sk] = sv
        if smooth:
            if self.smooth_eval_return is None:
                self.smooth_eval_return = log_info['eval/return/mean']
            else:
                self.smooth_eval_return = self.smooth_eval_return * self.smooth_tau
                self.smooth_eval_return += (1 - self.smooth_tau) * log_info['eval/return/mean']
            log_info['eval/smooth_return/mean'] = self.smooth_eval_return
            if self.smooth_eval_return > self._best_eval_ret:
                self._eval_is_best = True
                self._best_eval_ret = self.smooth_eval_return
            else:
                self._eval_is_best = False
        return log_info, raw_traj_info

    def rollout_once(self, *args, **kwargs):
        t0 = time.perf_counter()
        self.agent.eval_mode()
        traj = self.runner(**kwargs)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        return traj, elapsed_time

    def train_once(self):
        self.optim_stime = time.perf_counter()
        optim_infos = []
        for oe in range(sac_cfg.opt_num):
            sampled_data = self.memory.sample(batch_size=sac_cfg.batch_size)
            sampled_data = Trajectory(traj_data=sampled_data)
            batch_data = dict(
                obs=sampled_data.obs,
                next_obs=sampled_data.next_obs,
                actions=sampled_data.actions,
                dones=sampled_data.dones,
                rewards=sampled_data.rewards
            )
            optim_info = self.agent.optimize(batch_data)
            optim_infos.append(optim_info)
        return self.get_train_log(optim_infos)

    def get_train_log(self, optim_infos):
        log_info = dict()
        for key in optim_infos[0].keys():
            if 'val' in key:
                continue
            log_info[key] = np.mean([inf[key] for inf in optim_infos if key in inf])
      
        for key in ['q1_val', 'q2_val']:
            k_stats = get_list_stats([inf[key] for inf in optim_infos if key in inf])
            for sk, sv in k_stats.items():
                log_info[f'{key}/' + sk] = sv

        t1 = time.perf_counter()
        log_info['optim_time'] = t1 - self.optim_stime
        train_log_info = dict()
        for key, val in log_info.items():
            train_log_info['train/' + key] = val
        return train_log_info

    def add_traj_to_memory(self, traj):
        for sd in traj.traj_data:
            self.memory.append(deepcopy(sd))
        self.cur_step += traj.total_steps
