#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
#python ppo.py --env_id='Hopper-v2' &
#python ppo.py --env_id='Hopper-v2' --seed=234 &
#python ppo.py --env_id='Walker2d-v2' &
#python ppo.py --env_id='Walker2d-v2' --seed=234 &
#python ppo.py --env_id='Ant-v2' --seed=234 &
#wait
KMP_INIT_AT_FORK=FALSE python ppo.py --env_id='Swimmer-v2' &
KMP_INIT_AT_FORK=FALSE python ppo.py --env_id='Swimmer-v2' --seed=234 &
KMP_INIT_AT_FORK=FALSE python ppo.py --env_id='HalfCheetah-v2' &
KMP_INIT_AT_FORK=FALSE python ppo.py --env_id='HalfCheetah-v2' --seed=234 &
#python ppo.py --env_id='Ant-v2' &
wait
