from dataclasses import dataclass
from typing import Any

from easyrl.configs.ppo_config import PPOConfig
from easyrl.configs.sac_config import SACConfig
from easyrl.utils.rl_logger import logger


@dataclass
class CFG:
    alg: Any = None


cfg = CFG()


def set_config(alg):
    global cfg
    if alg == 'ppo':
        cfg.alg = PPOConfig()
    elif alg == 'sac':
        cfg.alg = SACConfig()
    elif alg == 'sac_adv':
        cfg.alg = SACAdvConfig()
    elif alg == 'redq':
        cfg.alg = REQDConfig()
    elif alg == 'offppo':
        cfg.alg = OffPPOConfig()
    else:
        raise ValueError(f'Unimplemented algorithm: {alg}')
    logger.info(f'Alogrithm type:{type(cfg.alg)}')
