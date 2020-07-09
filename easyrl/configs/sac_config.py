from dataclasses import dataclass

from easyrl.configs.basic_config import BasicConfig


@dataclass
class SACConfig(BasicConfig):
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    use_amsgrad: bool = True


sac_cfg = SACConfig()
