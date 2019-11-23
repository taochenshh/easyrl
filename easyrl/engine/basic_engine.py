class BasicEngine:
    def __init__(self, agent, env, runner, **kwargs):
        self.agent = agent
        self.env = env
        self.runner = runner

    def train(self, **kwargs):
        raise NotImplementedError

    def eval(self, **kwargs):
        raise NotImplementedError
