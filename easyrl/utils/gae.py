import numpy as np


def get_gae(gamma, lam, rewards, value_estimates, value_next):
    mb_advs = np.zeros_like(rewards)
    lastgaelam = 0
    value_estimates = np.concatenate((value_estimates, value_next), axis=0)
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * value_estimates[t + 1] - value_estimates[t]
        mb_advs[t] = lastgaelam = delta + gamma * lam * lastgaelam
    mb_returns = mb_advs + value_estimates[:-1, ...]
    return mb_advs, mb_returns
