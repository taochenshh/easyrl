import gym
import torch.nn as nn

from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.configs.ppo_config import ppo_cfg
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.episodic_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env


def main():
    cfg_from_cmd(ppo_cfg)
    if ppo_cfg.resume or ppo_cfg.test:
        ppo_cfg.restore_cfg()
    if ppo_cfg.env_id is None:
        ppo_cfg.env_id = 'Hopper-v2'
    set_random_seed(ppo_cfg.seed)
    env = make_vec_env(ppo_cfg.env_id, ppo_cfg.num_envs)
    env.reset()
    ob_size = env.observation_space.shape[0]

    actor_body = MLP(input_size=ob_size,
                     hidden_sizes=[64],
                     output_size=64,
                     hidden_act=nn.ReLU,
                     output_act=nn.ReLU)
    critic_body = MLP(input_size=ob_size,
                      hidden_sizes=[64],
                      output_size=64,
                      hidden_act=nn.ReLU,
                      output_act=nn.ReLU)
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_size = env.action_space.n
        actor = CategoricalPolicy(actor_body, action_dim=act_size)
    elif isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        actor = DiagGaussianPolicy(actor_body, action_dim=act_size,
                                   tanh_on_dist=ppo_cfg.tanh_on_dist,
                                   std_cond_in=ppo_cfg.std_cond_in)
    else:
        raise TypeError(f'Unknown action space '
                        f'type: {env.action_space}')

    critic = ValueNet(critic_body)
    agent = PPOAgent(actor, critic)
    runner = EpisodicRunner(agent=agent, env=env)
    engine = PPOEngine(agent=agent,
                       env=env,
                       runner=runner)
    if not ppo_cfg.test:
        engine.train()
    else:
        stat_info, raw_traj_info = engine.eval(render=ppo_cfg.render,
                                               save_eval_traj=ppo_cfg.save_test_traj,
                                               eval_num=ppo_cfg.test_num,
                                               sleep_time=0.04)
        import pprint
        pprint.pprint(stat_info)


if __name__ == '__main__':
    main()
