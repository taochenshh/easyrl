import torch
import torch.optim as optim

from easyrl.agents.base_agent import BaseAgent
from easyrl.configs.ppo_config import ppo_cfg
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import torch_to_np


class PPOAgent(BaseAgent):
    def __init__(self, actor, critic, same_body=False):
        self.actor = actor
        self.critic = critic
        self.actor.to(ppo_cfg.device)
        self.critic.to(ppo_cfg.device)
        self.same_body = same_body
        self.val_loss_criterion = nn.SmoothL1Loss().to(ppo_cfg.device)
        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.all_params = list(set(all_params))
        if self.same_body:
            self.optimizer = optim.Adam(self.all_params,
                                        lr=ppo_cfg.policy_lr,
                                        weight_decay=ppo_cfg.weight_decay,
                                        amsgrad=ppo_cfg.use_amsgrad)
        else:
            self.optimizer = optim.Adam([{'params': self.actor.parameters(),
                                          'lr': ppo_cfg.policy_lr},
                                         {'params': self.critic.parameters(),
                                          'lr': ppo_cfg.value_lr}],
                                        weight_decay=ppo_cfg.weight_decay,
                                        amsgrad=ppo_cfg.use_amsgrad
                                        )

    def get_action(self, ob, sample=True, **kwargs):
        t_ob = torch.from_numpy(ob).float().to(ppo_cfg.device)
        act_dist, val = self._get_act_val(t_ob)
        action = action_from_dist(act_dist,
                                  sample=sample)
        log_prob = action_log_prob(action, act_dist)
        entropy = act_dist.entropy()
        action_info = dict(
            log_prob=log_prob,
            entropy=entropy,
            val=val
        )
        return action, action_info

    def _get_act_val(self, ob):
        act_dist, body_out = self.actor(ob)
        if self.same_body:
            val = self.critic(body_x=body_out)
        else:
            val = self.critic(x=ob)
        return act_dist, val

    def optimize(self, data, **kwargs):
        for key, val in data.items():
            data[key] = torch.from_numpy(val).float().to(ppo_cfg.device)
        ob = data['ob']
        action = data['action']
        ret = data['return']
        adv = data['adv']
        old_log_prob = data['log_prob']
        old_val = data['val']

        if ppo_cfg.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        act_dist, val = self._get_act_val(ob)
        log_prob = action_log_prob(action, act_dist)
        entropy = act_dist.entropy()
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
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio,
                                      1 - ppo_cfg.clip_range,
                                      1 + ppo_cfg.clip_range)
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = torch.mean(pg_loss)

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
            pg_loss=pg_loss.item,
            vf_loss=vf_loss.item,
            total_loss=loss.item,
            entropy=entropy.item,
            approx_kl=approx_kl.item,
            clip_frac=clip_frac
        )
        if grad_norm is not None:
            optim_info['grad_norm'] = grad_norm

    def save_model(self):
        ckpt_file = os.path.join(self.model_dir,
                                 'ckpt_{:08d}.pth'.format(step))
        color_print.print_yellow('Saving checkpoint: %s' % ckpt_file)
        data_to_save = {
            'ckpt_step': step,
            'global_ep': self.global_ep,
            'p_net_state_dict': self.p_net.state_dict(),
            'q_net1_state_dict': self.q_net1.state_dict(),
            'q_net2_state_dict': self.q_net2.state_dict(),
        }
        if is_best:
            torch.save(data_to_save, os.path.join(self.model_dir,
                                                  'model_best.pth'))
