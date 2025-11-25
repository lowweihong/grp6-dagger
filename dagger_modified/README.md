# Modified DAgger Implementation

This directory contains a modified version of the DAgger algorithm with two key enhancements:

## Modifications

### 1. KDE-based Trajectory Filtering
- **Purpose**: Prevent adding redundant or similar trajectories to the dataset
- **Method**: Uses Kernel Density Estimation (KDE) to estimate how densely the existing dataset covers different regions of the state space
- **Process**:
  - For each new trajectory, compute its density relative to the existing dataset
  - If density is **below threshold**: trajectory explores new regions → **ADD** to dataset
  - If density is **above threshold**: trajectory is redundant → **SKIP**
- **Benefits**: 
  - Better dataset coverage
  - More efficient use of training data
  - Improved generalization

### 2. Action Noise During Sampling
- **Purpose**: Promote exploration and improve robustness
- **Method**: With probability α, add Gaussian noise to actions: `a' = π(s) + n` where `n ~ N(0, σ²)`
- **Process**:
  - After selecting policy action (expert or student)
  - With probability `ACTION_NOISE_PROB` (α), add noise
  - Clip resulting action to valid bounds: steering [-1, 1], gas/brake [0, 1]
- **Benefits**:
  - Better exploration of state space
  - More robust policy learning
  - Helps escape local minima

## Hyperparameters

### KDE Filtering
- `--kde-bandwidth`: KDE bandwidth parameter (default: 0.1)
- `--kde-threshold`: Density threshold for filtering (default: 0.5)

### Action Noise
- `--action-noise-prob`: Probability of adding noise (α, default: 0.1)
- `--action-noise-std`: Standard deviation of noise (σ, default: 0.1)

## Usage

```bash
cd dagger_modified
python train_dagger_modified.py \
    --expert-model ../expert_implementations/logs/ppo/CarRacing-v3_6/best_model.zip \
    --iterations 20 \
    --kde-threshold 0.5 \
    --action-noise-prob 0.1 \
    --action-noise-std 0.1
```

## Dependencies

Additional dependencies required:
```bash
pip install scikit-learn
```

## Output

The implementation tracks:
- Standard DAgger metrics (reward, falls, etc.)
- **Trajectory filtering statistics**: acceptance rate, number of accepted/rejected trajectories
- All metrics logged to TensorBoard and saved to JSON

## Files

- `train_dagger_modified.py`: Main training script with both modifications
- `results/`: Directory for saved models, logs, and results

