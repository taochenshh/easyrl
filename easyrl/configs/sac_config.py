from dataclasses import dataclass

from easyrl.configs.basic_config import BasicConfig


@dataclass
class SACConfig(BasicConfig):
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    warmup_steps: int = 10000
    use_amsgrad: bool = True
    opt_interval: int = 50  # perform optimization every n environment steps
    opt_num: int = 50  # how many optimization loops in every optimization stage
    alpha: float = None
    rew_discount: float = 0.99
    replay_size: int = 1000000
    polyak: float = 0.995


sac_cfg = SACConfig()
