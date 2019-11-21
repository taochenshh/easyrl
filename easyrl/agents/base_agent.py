class BaseAgent:

    def get_action(self, ob, sample=True, **kwargs):
        raise NotImplementedError

    def optimize(self, data, **kwargs):
        raise NotImplementedError
