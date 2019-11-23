from easyrl.agents.ppo import PPOAgent
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.configs.ppo_config import ppo_cfg
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.episodic_runner import EpisodicRunner
from easyrl.utils.gym_util import make_vec_env


def main():
    cfg_from_cmd(ppo_cfg)
    if ppo_cfg.resume:
        ppo_cfg.restore_cfg()
    env = make_vec_env(ppo_cfg.env, 2)
    env.reset()
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
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
    actor = DiagGaussianPolicy(actor_body, action_dim=act_size)
    critic = ValueNet(critic_body)
    agent = PPOAgent(actor, critic)
    runner = EpisodicRunner(agent=agent, env=env)
    engine = PPOEngine(agent=agent,
                       env=env,
                       runner=runner)
    engine.train()


if __name__ == '__main__':
    main()
