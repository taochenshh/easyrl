import gym
import torch.nn as nn

from easyrl.agents.ppo_rnn_agent import PPORNNAgent
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.configs.ppo_config import ppo_cfg
from easyrl.engine.ppo_rnn_engine import PPORNNEngine
from easyrl.models.mlp import MLP
from easyrl.models.rnn_base import RNNBase
from easyrl.models.rnn_categorical_policy import RNNCategoricalPolicy
from easyrl.models.rnn_diag_gaussian_policy import RNNDiagGaussianPolicy
from easyrl.models.rnn_value_net import RNNValueNet
from easyrl.runner.rnn_runner import RNNRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env


def main():
    cfg_from_cmd(ppo_cfg)
    if ppo_cfg.resume or ppo_cfg.test:
        if ppo_cfg.test:
            skip_params = [
                'test_num',
                'num_envs',
                'sample_action',
                'seed'
            ]
        else:
            skip_params = []
        ppo_cfg.restore_cfg(skip_params=skip_params)
    if ppo_cfg.env_name is None:
        ppo_cfg.env_name = 'Hopper-v2'
    set_random_seed(ppo_cfg.seed)
    env = make_vec_env(ppo_cfg.env_name,
                       ppo_cfg.num_envs,
                       seed=ppo_cfg.seed)
    env.reset()
    ob_size = env.observation_space.shape[0]

    actor_body = MLP(input_size=ob_size,
                     hidden_sizes=[256],
                     output_size=256,
                     hidden_act=nn.ReLU,
                     output_act=nn.ReLU)
    actor_body = RNNBase(body_net=actor_body,
                         rnn_features=256,
                         in_features=256,
                         rnn_layers=1,
                         )
    critic_body = MLP(input_size=ob_size,
                      hidden_sizes=[256],
                      output_size=256,
                      hidden_act=nn.ReLU,
                      output_act=nn.ReLU)
    critic_body = RNNBase(body_net=critic_body,
                          rnn_features=256,
                          in_features=256,
                          rnn_layers=1,
                          )
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_size = env.action_space.n
        actor = RNNCategoricalPolicy(actor_body, action_dim=act_size)
    elif isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        actor = RNNDiagGaussianPolicy(actor_body, action_dim=act_size,
                                      tanh_on_dist=ppo_cfg.tanh_on_dist,
                                      std_cond_in=ppo_cfg.std_cond_in)
    else:
        raise TypeError(f'Unknown action space '
                        f'type: {env.action_space}')

    critic = RNNValueNet(critic_body)
    agent = PPORNNAgent(actor, critic)
    runner = RNNRunner(agent=agent, env=env)
    engine = PPORNNEngine(agent=agent,
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
    env.close()


if __name__ == '__main__':
    main()
