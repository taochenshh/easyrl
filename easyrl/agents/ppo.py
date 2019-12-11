from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from easyrl.agents.base_agent import BaseAgent
from easyrl.configs.ppo_config import ppo_cfg
from easyrl.utils.common import linear_decay_percent
from easyrl.utils.rl_logger import logger
from easyrl.utils.torch_util import action_entropy
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import torch_to_np
from torch.optim.lr_scheduler import LambdaLR


class PPOAgent(BaseAgent):
    def __init__(self, actor, critic, same_body=False):
        self.actor = actor
        self.critic = critic
        self.actor.to(ppo_cfg.device)
        self.critic.to(ppo_cfg.device)
        self.same_body = same_body
        if ppo_cfg.vf_loss_type == 'mse':
            self.val_loss_criterion = nn.MSELoss().to(ppo_cfg.device)
        elif ppo_cfg.vf_loss_type == 'smoothl1':
            self.val_loss_criterion = nn.SmoothL1Loss().to(ppo_cfg.device)
        else:
            raise TypeError(f'Unknown value loss type: {ppo_cfg.vf_loss_type}!')
        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        # keep unique elements only. The following code works for python >=3.7
        # for earlier version of python, u need to use OrderedDict
        self.all_params = dict.fromkeys(all_params).keys()
        if ppo_cfg.max_steps > ppo_cfg.max_decay_steps:
            raise ValueError('max_steps should be greater than max_decay_steps.')
        total_epochs = int(np.ceil(ppo_cfg.max_decay_steps / (ppo_cfg.num_envs *
                                                              ppo_cfg.episode_steps)))
        if ppo_cfg.linear_decay_clip_range:
            self.clip_range_decay_rate = ppo_cfg.clip_range / float(total_epochs)
        p_lr_lambda = partial(linear_decay_percent,
                              total_epochs=total_epochs)

        if self.same_body:
            self.optimizer = optim.Adam(self.all_params,
                                        lr=ppo_cfg.policy_lr,
                                        weight_decay=ppo_cfg.weight_decay,
                                        amsgrad=ppo_cfg.use_amsgrad)
            self.lr_scheduler = LambdaLR(optimizer=self.optimizer,
                                         lr_lambda=[p_lr_lambda])
        else:
            self.optimizer = optim.Adam([{'params': self.actor.parameters(),
                                          'lr': ppo_cfg.policy_lr},
                                         {'params': self.critic.parameters(),
                                          'lr': ppo_cfg.value_lr}],
                                        weight_decay=ppo_cfg.weight_decay,
                                        amsgrad=ppo_cfg.use_amsgrad
                                        )

            v_lr_lambda = partial(linear_decay_percent,
                                  total_epochs=total_epochs)
            self.lr_scheduler = LambdaLR(optimizer=self.optimizer,
                                         lr_lambda=[p_lr_lambda, v_lr_lambda])

    @torch.no_grad()
    def get_action(self, ob, sample=True, **kwargs):
        self.eval_mode()
        t_ob = torch.from_numpy(ob).float().to(ppo_cfg.device)
        act_dist, val = self.get_act_val(t_ob)
        action = action_from_dist(act_dist,
                                  sample=sample)
        log_prob = action_log_prob(action, act_dist)
        entropy = action_entropy(act_dist, log_prob)
        action_info = dict(
            log_prob=torch_to_np(log_prob),
            entropy=torch_to_np(entropy),
            val=torch_to_np(val)
        )
        return torch_to_np(action), action_info

    def get_act_val(self, ob):
        if isinstance(ob, np.ndarray):
            ob = torch.from_numpy(ob).float().to(ppo_cfg.device)
        act_dist, body_out = self.actor(ob)
        if self.same_body:
            val = self.critic(body_x=body_out)
        else:
            val = self.critic(x=ob)
        val = val.squeeze(-1)
        return act_dist, val

    def optimize(self, data, **kwargs):
        self.train_mode()
        for key, val in data.items():
            data[key] = val.float().to(ppo_cfg.device)
        ob = data['ob']
        action = data['action']
        ret = data['ret']
        adv = data['adv']
        old_log_prob = data['log_prob']
        old_val = data['val']

        act_dist, val = self.get_act_val(ob)
        log_prob = action_log_prob(action, act_dist)
        entropy = action_entropy(act_dist, log_prob)
        if not all([x.ndim == 1 for x in [val, entropy, log_prob]]):
            raise ValueError('val, entropy, log_prob should be 1-dim!')

        if ppo_cfg.clip_vf_loss:
            clipped_val = old_val + torch.clamp(val - old_val,
                                                -ppo_cfg.clip_range,
                                                ppo_cfg.clip_range)
            vf_loss1 = torch.pow(val - ret, 2)
            vf_loss2 = torch.pow(clipped_val - ret, 2)
            vf_loss = 0.5 * torch.mean(torch.max(vf_loss1,
                                                 vf_loss2))
        else:
            val = torch.squeeze(val)
            vf_loss = 0.5 * self.val_loss_criterion(val, ret)

        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = adv * ratio
        surr2 = adv * torch.clamp(ratio,
                                  1 - ppo_cfg.clip_range,
                                  1 + ppo_cfg.clip_range)
        pg_loss = -torch.mean(torch.min(surr1, surr2))

        entropy = torch.mean(entropy)
        loss = pg_loss - entropy * ppo_cfg.ent_coef + \
               vf_loss * ppo_cfg.vf_coef
        with torch.no_grad():
            approx_kl = 0.5 * torch.mean(torch.pow(old_log_prob - log_prob, 2))
            clip_frac = np.mean(np.abs(torch_to_np(ratio) - 1.0) > ppo_cfg.clip_range)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = None
        if ppo_cfg.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.all_params,
                                                       ppo_cfg.max_grad_norm)
        self.optimizer.step()
        optim_info = dict(
            pg_loss=pg_loss.item(),
            vf_loss=vf_loss.item(),
            total_loss=loss.item(),
            entropy=entropy.item(),
            approx_kl=approx_kl.item(),
            clip_frac=clip_frac
        )
        if grad_norm is not None:
            optim_info['grad_norm'] = grad_norm
        return optim_info

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def decay_lr(self):
        self.lr_scheduler.step()

    def get_lr(self):
        cur_lr = self.lr_scheduler.get_lr()
        lrs = {'policy_lr': cur_lr[0]}
        if len(cur_lr) > 1:
            lrs['value_lr'] = cur_lr[1]
        return lrs

    def decay_clip_range(self):
        ppo_cfg.clip_range -= self.clip_range_decay_rate

    def save_model(self, is_best=False, step=None):
        if not ppo_cfg.save_best_only and step is not None:
            ckpt_file = ppo_cfg.model_dir \
                .joinpath('ckpt_{:012d}.pt'.format(step))
        else:
            ckpt_file = None
        if is_best:
            best_model_file = ppo_cfg.model_dir \
                .joinpath('model_best.pt')
        else:
            best_model_file = None

        if not ppo_cfg.save_best_only:
            saved_model_files = sorted(ppo_cfg.model_dir.glob('*.pt'))
            if len(saved_model_files) > ppo_cfg.max_saved_models:
                saved_model_files[0].unlink()

        data_to_save = {
            'step': step,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }
        if ppo_cfg.linear_decay_clip_range:
            data_to_save['clip_range'] = ppo_cfg.clip_range
            data_to_save['clip_range_decay_rate'] = self.clip_range_decay_rate
        logger.info(f'Exploration steps: {step}')
        for fl in [ckpt_file, best_model_file]:
            if fl is not None:
                logger.info(f'Saving checkpoint: {fl}.')
                torch.save(data_to_save, fl)

    def load_model(self, step=None):
        if step is None:
            ckpt_file = Path(ppo_cfg.model_dir) \
                .joinpath('model_best.pt')
        else:
            ckpt_file = Path(ppo_cfg.model_dir) \
                .joinpath('ckpt_{:012d}.pt'.format(step))
        logger.info(f'Loading model from {ckpt_file}')
        if not ckpt_file.exists():
            raise ValueError(f'Checkpoint file ({ckpt_file}) '
                             f'does not exist!')
        ckpt_data = torch.load(ckpt_file)
        self.actor.load_state_dict(ckpt_data['actor_state_dict'])
        self.critic.load_state_dict(ckpt_data['critic_state_dict'])
        self.optimizer.load_state_dict(ckpt_data['optim_state_dict'])
        self.lr_scheduler.load_state_dict(ckpt_data['lr_scheduler_state_dict'])
        if ppo_cfg.linear_decay_clip_range:
            self.clip_range_decay_rate = ckpt_data['clip_range_decay_rate']
            ppo_cfg.clip_range = ckpt_data['clip_range']
        return ckpt_data['step']
