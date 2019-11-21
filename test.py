import numpy as np
from IPython import embed

from easyrl.utils.gym_util import make_vec_env

env = make_vec_env('Hopper-v2', 2)
env.reset()

obs = []
rews = []
dones = []
infos = []
for i in range(30):
    ob, rew, done, info = env.step(np.array([[0.5, 0.5, 0.5] for i in range(2)]))
    obs.append(ob)
    rews.append(rew)
    dones.append(done)
    infos.append(info)

embed()
