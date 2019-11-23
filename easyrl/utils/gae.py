import numpy as np


def cal_gae(gamma, lam, rewards, value_estimates, last_value, dones):
    advs = np.zeros_like(rewards)
    last_gae_lam = 0
    value_estimates = np.concatenate((value_estimates,
                                      last_value),
                                     axis=0)
    for t in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * value_estimates[t + 1] * non_terminal - value_estimates[t]
        last_gae_lam = delta + gamma * lam * non_terminal * last_gae_lam
        advs[t] = last_gae_lam.copy()
    return advs
