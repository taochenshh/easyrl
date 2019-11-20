from easyrl.runner.base import BasicRunner
from easyrl.utils.data import Trajectory
from easyrl.utils.data import StepData


class EpisodicRunner(BasicRunner):
    def __init__(self, agent, env):
        super().__init__(agent=agent,
                         env=env)

    def __call__(self, time_steps, deterministic=False, return_on_done=False, **kwargs):
        traj = Trajectory()
        ob = self.env.reset()
        for t in range(time_steps):
            action = self.agent.get_action(ob,
                                           deterministic=deterministic)
            ob, reward, done, info = self.env.step(action)










