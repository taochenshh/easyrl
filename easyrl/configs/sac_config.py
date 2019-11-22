from easyrl.configs.basic_config import BasicConfig
from dataclasses import dataclass


@dataclass
class SACConfig(BasicConfig):
    actor_lr: float = 1e-4
