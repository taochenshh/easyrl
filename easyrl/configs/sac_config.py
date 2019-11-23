from dataclasses import dataclass

from easyrl.configs.basic_config import BasicConfig


@dataclass
class SACConfig(BasicConfig):
    actor_lr: float = 1e-4
