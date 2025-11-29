# Modified DAgger Implementation with KDE Filtering and Action Noise

This directory contains a modified version of the DAgger algorithm with two key enhancements:

## Modifications

### 1. KDE-based Trajectory Filtering
Instead of adding all sampled trajectories to the dataset, we use Kernel Density Estimation (KDE) to filter trajectories based on how well-explored the state space region is.

- **How it works**: For each new trajectory, we compute the density of the existing dataset around that trajectory using KDE. If the density is above a threshold, the region is well-explored and we filter out the trajectory. If the density is below the threshold, we keep it as it represents an underexplored region.

- **Benefits**: 
  - Prevents redundant data from well-explored regions
  - Focuses dataset on underexplored areas
  - Improves generalization by ensuring better state space coverage

### 2. Action Noise
Extends the expert mixing framework by adding Gaussian noise to student policy actions with probability α.

- **How it works**: With probability α, instead of using the student policy action directly, we add Gaussian noise: `a' = π(s) + n` where `n ~ N(0, σ²)`. The noisy action is clipped to valid action bounds.

- **Benefits**:
  - Promotes exploration during data collection
  - Improves robustness by exposing the model to noisy actions
  - Helps discover new state trajectories

## Hyperparameters

### Original DAgger Parameters
- `--expert-mixing-prob` (β): Probability of using expert policy (default: 0.1)
- Standard DAgger parameters (iterations, episodes, epochs, etc.)

### New Parameters
- `--action-noise-prob` (α): Probability of adding noise to student actions (default: 0.2)
- `--action-noise-std` (σ): Standard deviation of Gaussian noise (default: 0.1)
- `--kde-threshold`: Density threshold - keep trajectories below this (default: 0.1)
- `--kde-bandwidth`: Bandwidth for KDE kernel (default: 0.5)
- `--kde-sample-size`: Max samples to use for KDE estimation (default: 1000)

## Usage

```bash
python train_dagger_kde.py \
    --expert-model ../expert_implementations/logs/ppo/CarRacing-v3_6/best_model.zip \
    --iterations 20 \
    --episodes-per-iter 1 \
    --action-noise-prob 0.2 \
    --action-noise-std 0.1 \
    --kde-threshold 0.1 \
    --kde-bandwidth 0.5
```

## Output

The script generates:
- Trained student model in `results/`
- Performance logs with filtering statistics
- JSON results with detailed metrics
- TensorBoard logs (if enabled)

## Key Differences from Original DAgger

1. **Data Collection**: Action selection includes noise with probability α
2. **Data Filtering**: Trajectories are filtered using KDE before adding to dataset
3. **Metrics**: Additional logging for filtering statistics (filter ratio, density metrics)
4. **Efficiency**: KDE uses subsampling for large datasets

## Dependencies

Requires `scikit-learn` for KDE:
```bash
pip install scikit-learn
```

