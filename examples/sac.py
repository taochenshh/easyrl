import gym
import torch.nn as nn
import torch
from easyrl.agents.sac_agent import SACAgent
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.configs.sac_config import sac_cfg
from easyrl.engine.sac_engine import SACEngine
from easyrl.replays.circular_buffer import CyclicBuffer
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.step_runner import StepRunner
from easyrl.utils.common import set_random_seed


def main():
    torch.set_num_threads(1)
    cfg_from_cmd(sac_cfg)
    if sac_cfg.resume or sac_cfg.test:
        if sac_cfg.test:
            skip_params = [
                'test_num',
                'num_envs',
                'sample_action',
            ]
        else:
            skip_params = []
        sac_cfg.restore_cfg(skip_params=skip_params)
    if sac_cfg.env_name is None:
        sac_cfg.env_name = 'HalfCheetah-v2'
    if not sac_cfg.test:
        sac_cfg.test_num = 10
    set_random_seed(sac_cfg.seed)
    env = gym.make(sac_cfg.env_name)
    env.seed(sac_cfg.seed)
    eval_env = gym.make(sac_cfg.env_name)
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    actor_body = MLP(input_size=ob_size,
                     hidden_sizes=[256],
                     output_size=256,
                     hidden_act=nn.ReLU,
                     output_act=nn.ReLU)
    q1_body = MLP(input_size=ob_size + act_size,
                  hidden_sizes=[256],
                  output_size=256,
                  hidden_act=nn.ReLU,
                  output_act=nn.ReLU)
    q2_body = MLP(input_size=ob_size + act_size,
                  hidden_sizes=[256],
                  output_size=256,
                  hidden_act=nn.ReLU,
                  output_act=nn.ReLU)
    actor = DiagGaussianPolicy(actor_body, action_dim=act_size,
                               tanh_on_dist=True,
                               std_cond_in=True,
                               clamp_log_std=True)
    q1 = ValueNet(q1_body)
    q2 = ValueNet(q2_body)
    memory = CyclicBuffer(capacity=sac_cfg.replay_size)
    agent = SACAgent(actor, q1=q1, q2=q2, env=env, memory=memory)
    runner = StepRunner(agent=agent, env=env, eval_env=eval_env)
    
    engine = SACEngine(agent=agent,
                       runner=runner)
    if not sac_cfg.test:
        engine.train()
    else:
        stat_info, raw_traj_info = engine.eval(render=sac_cfg.render,
                                               save_eval_traj=sac_cfg.save_test_traj,
                                               eval_num=sac_cfg.test_num,
                                               sleep_time=0.04)
        import pprint
        pprint.pprint(stat_info)
    env.close()


if __name__ == '__main__':
    main()
