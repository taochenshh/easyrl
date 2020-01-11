class BasicRunner:
    def __init__(self, agent, env, eval_env=None):
        self.agent = agent
        self.train_env = env
        self.eval_env = env if eval_env is None else eval_env

    def __call__(self, **kwargs):
        raise NotImplementedError
