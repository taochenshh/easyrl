import gym
import torch.nn as nn

from easyrl.agents.ppo_rnn_agent import PPORNNAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.configs.command_line import cfg_from_cmd
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
    set_config('ppo')
    cfg_from_cmd(cfg.alg)
    if cfg.alg.resume or cfg.alg.test:
        if cfg.alg.test:
            skip_params = [
                'test_num',
                'num_envs',
                'sample_action',
            ]
        else:
            skip_params = []
        cfg.alg.restore_cfg(skip_params=skip_params)
    if cfg.alg.env_name is None:
        cfg.alg.env_name = 'HalfCheetah-v2'
    set_random_seed(cfg.alg.seed)
    env = make_vec_env(cfg.alg.env_name,
                       cfg.alg.num_envs,
                       seed=cfg.alg.seed)
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
                                      tanh_on_dist=cfg.alg.tanh_on_dist,
                                      std_cond_in=cfg.alg.std_cond_in)
    else:
        raise TypeError(f'Unknown action space '
                        f'type: {env.action_space}')

    critic = RNNValueNet(critic_body)
    agent = PPORNNAgent(actor, critic)
    runner = RNNRunner(agent=agent, env=env)
    engine = PPORNNEngine(agent=agent,
                          runner=runner)
    if not cfg.alg.test:
        engine.train()
    else:
        stat_info, raw_traj_info = engine.eval(render=cfg.alg.render,
                                               save_eval_traj=cfg.alg.save_test_traj,
                                               eval_num=cfg.alg.test_num,
                                               sleep_time=0.04)
        import pprint
        pprint.pprint(stat_info)
    env.close()


if __name__ == '__main__':
    main()
