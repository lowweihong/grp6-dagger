# grp6-dagger

## Train expert with `rl-baselines3-zoo`
> git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
> python rl-baselines3-zoo/train.py \
    --algo ppo \
    --env CarRacing-v3 \
    --device cuda \
    --n-timesteps 275000 \
    --tensorboard-log ./ppo-carracing >> log_ppo_carracing.log 2>&1&

