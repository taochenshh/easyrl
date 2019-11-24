#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
#python ppo.py --env_id='Hopper-v2' &
#python ppo.py --env_id='Hopper-v2' --seed=234 &
#python ppo.py --env_id='Walker2d-v2' &
#python ppo.py --env_id='Walker2d-v2' --seed=234 &
#python ppo.py --env_id='Ant-v2' --seed=234 &
#wait
#KMP_INIT_AT_FORK=FALSE python ppo.py --env_id='Swimmer-v2' &
#KMP_INIT_AT_FORK=FALSE python ppo.py --env_id='Swimmer-v2' --seed=234 &
#KMP_INIT_AT_FORK=FALSE python ppo.py --env_id='HalfCheetah-v2' &
#KMP_INIT_AT_FORK=FALSE python ppo.py --env_id='HalfCheetah-v2' --seed=234 &
#KMP_INIT_AT_FORK=FALSE python ppo_tanh.py --env_id='Swimmer-v2' --save_dir='tanh_data' &
#KMP_INIT_AT_FORK=FALSE python ppo_tanh.py --env_id='Swimmer-v2' --seed=234 --save_dir='tanh_data' &
#KMP_INIT_AT_FORK=FALSE python ppo_tanh.py --env_id='HalfCheetah-v2' --save_dir='tanh_data' &
#KMP_INIT_AT_FORK=FALSE python ppo_tanh.py --env_id='HalfCheetah-v2' --seed=234 --save_dir='tanh_data' &
KMP_INIT_AT_FORK=FALSE python ppo_tanh.py --env_id='Swimmer-v2' --ent_coef=0 --save_dir='tanh_data/normal_init' &
KMP_INIT_AT_FORK=FALSE python ppo_tanh.py --env_id='Swimmer-v2' --seed=234 --ent_coef=0 --save_dir='tanh_data/normal_init' &
KMP_INIT_AT_FORK=FALSE python ppo_tanh.py --env_id='HalfCheetah-v2' --ent_coef=0 --save_dir='tanh_data/normal_init' &
KMP_INIT_AT_FORK=FALSE python ppo_tanh.py --env_id='HalfCheetah-v2' --ent_coef=0 --seed=234 --save_dir='tanh_data/normal_init' &
#python ppo.py --env_id='Ant-v2' &
wait
