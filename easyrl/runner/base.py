class BasicRunner:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def __call__(self, **kwargs):
        raise NotImplementedError
