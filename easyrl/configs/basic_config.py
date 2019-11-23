import json
import shutil
from pathlib import Path

from dataclasses import asdict
from dataclasses import dataclass

from easyrl.utils.rl_logger import logger
from easyrl.utils.common import get_git_infos

@dataclass
class BasicConfig:
    env_id: str = 'Hopper-v2'
    seed: int = 1
    device: str = 'cuda'
    save_dir: str = 'data'
    eval_interval: int = 50000
    log_interval: int = 10000
    weight_decay: float = 0.005
    max_grad_norm: float = None
    batch_size: int = 64
    save_best_only: bool = True
    test: bool = False
    resume: bool = False
    resume_step: int = None

    @property
    def root_dir(self):
        return Path(__file__).resolve().parents[2]

    @property
    def data_dir(self):
        return Path.cwd().joinpath(self.save_dir)

    @property
    def model_dir(self):
        return self.data_dir.joinpath('model')

    @property
    def log_dir(self):
        return self.data_dir.joinpath('log')

    def create_model_log_dir(self):
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)
        Path.mkdir(self.model_dir, parents=True)
        Path.mkdir(self.log_dir, parents=True)
        hp_file = self.data_dir.joinpath('hp.json')
        hps = asdict(self)
        hps['git_info'] = get_git_infos(self.root_dir)
        with hp_file.open('w') as f:
            json.dump(hps, f, indent=2)

    def restore_cfg(self):
        hp_file = self.data_dir.joinpath('hp.json')
        with hp_file.open() as f:
            cfg_stored = json.load(f)
        for key, val in cfg_stored.items():
            if hasattr(self, key):
                setattr(self, key, val)
                logger.info(f'Restoring {key} to {val}.')
