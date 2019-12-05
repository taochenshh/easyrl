from dataclasses import dataclass

from easyrl.configs.basic_config import BasicConfig


@dataclass
class PPOConfig(BasicConfig):
    # if the actor and critic share body, then optimizer
    # will use policy_lr by default
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    linear_decay_lr: bool = True
    num_envs: int = 8
    opt_epochs: int = 10
    normalize_adv: bool = True
    clip_vf_loss: bool = False
    vf_loss_type: str = 'mse'
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    clip_range: float = 0.2
    linear_decay_clip_range: bool = True
    gae_lambda: float = 0.95
    rew_discount: float = 0.99
    max_steps: int = 1e6
    episode_steps: int = 1000
    use_amsgrad: bool = True
    tanh_on_dist: bool = False
    std_cond_in: bool = False


ppo_cfg = PPOConfig()
