import gym

from easyrl.envs.dummy_vec_env import DummyVecEnv
from easyrl.envs.shmem_vec_env import ShmemVecEnv
from easyrl.envs.timeout import TimeOutEnv
from easyrl.utils.rl_logger import logger


def make_vec_env(env_id, num_envs, seed=1, no_timeout=True, env_kwargs=None):
    logger.info(f'Creating {num_envs} environments.')
    if env_kwargs is None:
        env_kwargs = {}

    def make_env(env_id, rank, seed, no_timeout, env_kwargs):
        def _thunk():
            env = gym.make(env_id, **env_kwargs)
            if no_timeout:
                env = TimeOutEnv(env)
            env.seed(seed + rank)
            return env

        return _thunk

    envs = [make_env(env_id,
                     idx,
                     seed,
                     no_timeout,
                     env_kwargs) for idx in range(num_envs)]
    if num_envs > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)
    return envs
