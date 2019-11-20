from easyrl.utils.gym_util import make_vec_env

env = make_vec_env('Hopper-v2', 8)
env.reset()
from IPython import embed
embed()