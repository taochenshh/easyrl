python ppo.py --env_id='Hopper-v2' &
python ppo.py --env_id='Hopper-v2' --seed=234 &
python ppo.py --env_id='Walker2d-v2' &
python ppo.py --env_id='Walker2d-v2' --seed=234 &
python ppo.py --env_id='Ant-v2' --seed=234 &
wait
python ppo.py --env_id='Swimmer-v2' &
python ppo.py --env_id='Swimmer-v2' --seed=234 &
python ppo.py --env_id='HalfCheetah-v2' &
python ppo.py --env_id='HalfCheetah-v2' --seed=234 &
python ppo.py --env_id='Ant-v2' &
