class BasicEngine:
    def __init__(self, agent, runner, **kwargs):
        self.agent = agent
        self.runner = runner

    def train(self, **kwargs):
        raise NotImplementedError

    def eval(self, **kwargs):
        raise NotImplementedError
