import pickle
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from easyrl.agents.base_agent import BaseAgent
from easyrl.configs.sac_config import sac_cfg
from easyrl.utils.common import load_from_pickle
from easyrl.utils.common import save_to_pickle
from easyrl.utils.gym_util import num_space_dim
from easyrl.utils.rl_logger import logger
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import clip_grad
from easyrl.utils.torch_util import freeze_model
from easyrl.utils.torch_util import load_ckpt_data
from easyrl.utils.torch_util import load_state_dict
from easyrl.utils.torch_util import move_to
from easyrl.utils.torch_util import save_model
from easyrl.utils.torch_util import soft_update
from easyrl.utils.torch_util import torch_float
from easyrl.utils.torch_util import torch_to_np
from easyrl.utils.torch_util import unfreeze_model


class SACAgent(BaseAgent):
    def __init__(self, actor, q1, q2, env, memory):
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.memory = memory
        self.q1_tgt = deepcopy(self.q1)
        self.q2_tgt = deepcopy(self.q2)
        freeze_model(self.q1_tgt)
        freeze_model(self.q2_tgt)
        self.q1_tgt.eval()
        self.q2_tgt.eval()

        move_to([self.actor, self.q1, self.q2, self.q1_tgt, self.q2_tgt],
                device=sac_cfg.device)
        self.mem_file = sac_cfg.model_dir.joinpath('mem.pkl')
        optim_args = dict(
            lr=sac_cfg.actor_lr,
            weight_decay=sac_cfg.weight_decay,
            amsgrad=sac_cfg.use_amsgrad
        )

        self.pi_optimizer = optim.Adam(self.actor.parameters(),
                                       **optim_args)
        q_params = list(self.q1.parameters()) + list(self.q2.parameters())
        # keep unique elements only.
        self.q_params = dict.fromkeys(q_params).keys()
        optim_args['lr'] = sac_cfg.critic_lr
        self.q_optimizer = optim.Adam(self.q_params, **optim_args)
        if sac_cfg.alpha is None:
            if sac_cfg.tgt_entropy is None:
                self.tgt_entropy = -float(num_space_dim(env.action_space))
            else:
                self.tgt_entropy = sac_cfg.tgt_entropy
            self.log_alpha = nn.Parameter(torch.zeros(1, device=sac_cfg.device))
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha],
                lr=sac_cfg.actor_lr,
            )

    @property
    def alpha(self):
        if sac_cfg.alpha is None:
            return self.log_alpha.exp().item()
        else:
            return sac_cfg.alpha

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

    def optimize(self, data, *args, **kwargs):
        self.train_mode()
        for key, val in data.items():
            data[key] = torch_float(val, device=sac_cfg.device)
        obs = data['obs'].squeeze(1)
        actions = data['actions'].squeeze(1)
        next_obs = data['next_obs'].squeeze(1)
        rewards = data['rewards']
        dones = data['dones']
        q_info = self.update_q(obs=obs,
                               actions=actions,
                               next_obs=next_obs,
                               rewards=rewards,
                               dones=dones)
        pi_info = self.update_pi(obs=obs)
        alpha_info = self.update_alpha(pi_info['pi_entropy'])
        optim_info = {**q_info, **pi_info, **alpha_info}
        optim_info['alpha'] = self.alpha
        if hasattr(self, 'log_alpha'):
            optim_info['log_alpha'] = self.log_alpha.item()

        soft_update(self.q1_tgt, self.q1, sac_cfg.polyak)
        soft_update(self.q2_tgt, self.q2, sac_cfg.polyak)
        return optim_info

    def update_q(self, obs, actions, next_obs, rewards, dones):
        q1 = self.q1((obs, actions))[0]
        q2 = self.q2((obs, actions))[0]
        with torch.no_grad():
            next_act_dist = self.actor(next_obs)[0]
            next_actions = action_from_dist(next_act_dist,
                                            sample=True)
            nlog_prob = action_log_prob(next_actions, next_act_dist).unsqueeze(-1)
            nq1_tgt_val = self.q1_tgt((next_obs, next_actions))[0]
            nq2_tgt_val = self.q2_tgt((next_obs, next_actions))[0]
            nq_tgt_val = torch.min(nq1_tgt_val, nq2_tgt_val) - self.alpha * nlog_prob
            q_tgt_val = rewards + sac_cfg.rew_discount * (1 - dones) * nq_tgt_val
        loss_q1 = F.mse_loss(q1, q_tgt_val)
        loss_q2 = F.mse_loss(q2, q_tgt_val)
        loss_q = loss_q1 + loss_q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        grad_norm = clip_grad(self.q_params, sac_cfg.max_grad_norm)
        self.q_optimizer.step()
        q_info = dict(
            q1_loss=loss_q1.item(),
            q2_loss=loss_q2.item(),
            q1_val=torch_to_np(q1),
            q2_val=torch_to_np(q2)
        )
        q_info['q_grad_norm'] = grad_norm
        return q_info

    def update_pi(self, obs):
        freeze_model([self.q1, self.q2])
        act_dist = self.actor(obs)[0]
        new_actions = action_from_dist(act_dist,
                                       sample=True)
        new_log_prob = action_log_prob(new_actions, act_dist).unsqueeze(-1)
        new_q1 = self.q1((obs, new_actions))[0]
        new_q2 = self.q2((obs, new_actions))[0]
        new_q = torch.min(new_q1, new_q2)

        loss_pi = (self.alpha * new_log_prob - new_q).mean()
        self.q_optimizer.zero_grad()
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        grad_norm = clip_grad(self.actor.parameters(), sac_cfg.max_grad_norm)
        self.pi_optimizer.step()
        pi_info = dict(
            pi_loss=loss_pi.item(),
            pi_entropy=-new_log_prob.mean().item()
        )
        pi_info['pi_grad_norm'] = grad_norm
        unfreeze_model([self.q1, self.q2])
        return pi_info

    def update_alpha(self, pi_entropy):
        if sac_cfg.alpha is not None:
            return dict()
        alpha_loss = self.log_alpha.exp() * (pi_entropy - self.tgt_entropy)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha_info = dict(
            alpha_loss=alpha_loss.item()
        )
        return alpha_info

    def train_mode(self):
        self.actor.train()
        self.q1.train()
        self.q2.train()

    def eval_mode(self):
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()

    def save_model(self, is_best=False, step=None):
        data_to_save = {
            'step': step,
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q1_tgt_state_dict': self.q1_tgt.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q2_tgt_state_dict': self.q2_tgt.state_dict(),
            'pi_optim_state_dict': self.pi_optimizer.state_dict(),
            'q_optim_state_dict': self.q_optimizer.state_dict(),
        }
        if sac_cfg.alpha is None:
            data_to_save['log_alpha'] = self.log_alpha
            data_to_save['alpha_optim_state_dict'] = self.alpha_optimizer.state_dict()
        save_model(data_to_save, sac_cfg, is_best=is_best, step=step)
        logger.info(f'Saving the replay buffer to: {self.mem_file}.')
        save_to_pickle(self.memory, self.mem_file)
        logger.info('The replay buffer is saved.')

    def load_model(self, step=None, pretrain_model=None):
        ckpt_data = load_ckpt_data(sac_cfg, step=step,
                                   pretrain_model=pretrain_model)
        load_state_dict(self.actor,
                        ckpt_data['actor_state_dict'])
        load_state_dict(self.q1,
                        ckpt_data['q1_state_dict'])
        load_state_dict(self.q1_tgt,
                        ckpt_data['q1_tgt_state_dict'])
        load_state_dict(self.q2,
                        ckpt_data['q2_state_dict'])
        load_state_dict(self.q2_tgt,
                        ckpt_data['q2_tgt_state_dict'])
        if sac_cfg.alpha is None:
            self.log_alpha = ckpt_data['log_alpha']
        if pretrain_model is not None:
            return
        self.pi_optimizer.load_state_dict(ckpt_data['pi_optim_state_dict'])
        self.q_optimizer.load_state_dict(ckpt_data['q_optim_state_dict'])
        if sac_cfg.alpha is None:
            self.alpha_optimizer.load_state_dict(ckpt_data['alpha_optim_state_dict'])

        logger.info(f'Loading the replay buffer from: {self.mem_file}.')
        if not self.mem_file.exists():
            logger.warning('The replay buffer file is not founded!')
        else:
            try:
                self.memory = load_from_pickle(self.mem_file)
            except pickle.UnpicklingError:
                logger.warning('The replay buffer file is corrupted, hence, not loaded!')

        return ckpt_data['step']
