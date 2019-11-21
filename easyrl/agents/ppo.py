from easyrl.agents.base_agent import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic