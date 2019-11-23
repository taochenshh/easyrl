import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BasicConfig:
    seed: int = 1
    device: str = 'cuda'
    save_dir: str = 'data'
    eval_interval: int = 50
    log_interval: int = 1
    weight_decay: float = 0.005
    max_grad_norm: float = None
    batch_size: int = 64
    save_best_only: bool = True

    @property
    def model_dir(self):
        return Path.cwd().joinpath(self.save_dir).joinpath('model')

    @property
    def log_dir(self):
        return Path.cwd().joinpath(self.save_dir).joinpath('log')

    def create_model_log_dir(self):
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)
        Path.mkdir(self.model_dir, parents=True)
        Path.mkdir(self.log_dir, parents=True)
