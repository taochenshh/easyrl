import json
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from easyrl.utils.common import get_git_infos
from easyrl.utils.rl_logger import logger


@dataclass
class BasicConfig:
    env_id: str = None
    seed: int = 1
    device: str = 'cuda'
    save_dir: str = 'data'
    eval_interval: int = 10
    log_interval: int = 2
    weight_decay: float = 0.00
    max_grad_norm: float = None
    batch_size: int = 32
    save_best_only: bool = True
    test: bool = False
    resume: bool = False
    resume_step: int = None
    render: bool = False

    @property
    def root_dir(self):
        return Path(__file__).resolve().parents[2]

    @property
    def data_dir(self):
        if hasattr(self, 'diff_cfg') and 'save_dir' in self.diff_cfg:
            # if 'save_dir' is given, then it will just
            # use it as the data dir
            save_dir = Path(self.save_dir)
            if save_dir.is_absolute():
                data_dir = save_dir
            else:
                data_dir = Path.cwd().joinpath(self.save_dir)
            return data_dir
        data_dir = Path.cwd().joinpath(self.save_dir).joinpath(self.env_id)
        skip_params = ['env_id',
                       'save_dir',
                       'resume',
                       'resume_step',
                       'test',
                       'save_best_only',
                       'log_interval',
                       'eval_interval',
                       'render']
        if hasattr(self, 'diff_cfg'):
            path_name = ''
            if 'test' in self.diff_cfg:
                skip_params.append('num_envs')
            for key, val in self.diff_cfg.items():
                if key in skip_params:
                    continue
                if not path_name:
                    path_name += f'{key}_{val}'
                else:
                    path_name += f'_{key}_{val}'
            data_dir = data_dir.joinpath(path_name)
        else:
            data_dir = data_dir.joinpath('default')
        return data_dir

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
        skip_params = ['resume',
                       'resume_step',
                       'render']
        for key, val in cfg_stored.items():
            if hasattr(self, key) and key not in skip_params:
                setattr(self, key, val)
                logger.info(f'Restoring {key} to {val}.')
