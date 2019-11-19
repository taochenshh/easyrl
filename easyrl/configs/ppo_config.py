from dataclasses import dataclass

@dataclass
class PPOConfig:
    seed: int = 1
    lr: float = 3e-4
    num_envs: int = 8
    batch_size: int = 64
    weight_decay: float = 0.005
    eval_interval: int = 50
    log_interval: int = 1
    max_steps: int = 5e6
    episode_steps: int = 1000

