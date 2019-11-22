from easyrl.configs.basic_config import BasicConfig
from dataclasses import dataclass


@dataclass
class PPOConfig(BasicConfig):
    # if the actor and critic share body, then optimizer
    # will use policy_lr by default
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    num_envs: int = 8
    opt_epochs: int = 5
    normalize_adv: bool = True
    clip_vf_loss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    rew_discount: float = 0.99
    max_steps: int = 5e6
    episode_steps: int = 1000
    use_amsgrad: bool = False


ppo_cfg = PPOConfig()
