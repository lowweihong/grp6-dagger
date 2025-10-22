# grp6-dagger

## Train expert with `rl-baselines3-zoo`
```bash
> cd expert_implementations
> git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
> python rl-baselines3-zoo/train.py \
    --algo ppo \
    --env CarRacing-v3 \
    --device cuda \
    --n-timesteps 275000 \
    --tensorboard-log ./ppo-carracing >> log_ppo_carracing.log 2>&1&
```
### Performance

### Results
```
=== PPO CarRacing-v3 Results ===
Timestamp: 2025-10-22 01:25:49
Average Reward: 635.14
Reward Std: 245.44
Min Reward: 292.37
Max Reward: 853.85

Episode Details:
Episode 1: Reward=759.21, Steps=1000
Episode 2: Reward=292.37, Steps=1000
Episode 3: Reward=853.85, Steps=1000

```

#### Expert Performance Videos

**Episode 0:**
expert_implementations/videos_20251022_012428/ppo_carracing-episode-0.mp4

**Episode 1:**
expert_implementations/videos_20251022_012428/ppo_carracing-episode-1.mp4

**Episode 2:**
expert_implementations/videos_20251022_012428/ppo_carracing-episode-2.mp4



## Train dagger
```bash
> cd dagger_implementations
> nohup python train_dagger_2.py >> log_dagger2.log 2>&1&
```
### Performance
Current performance can be found at `dagger_performance_20251022_010617.txt`


