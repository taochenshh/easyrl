from dataclasses import dataclass


@dataclass
class PPOConfig:
    seed: int = 1
    device: str = 'cuda'
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    num_envs: int = 8
    batch_size: int = 64
    weight_decay: float = 0.005
    max_grad_norm: float = 1.0
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    rew_discount: float = 0.99
    eval_interval: int = 50
    log_interval: int = 1
    max_steps: int = 5e6
    episode_steps: int = 1000
