# DAgger Paper Reimplementation: A Comprehensive Study of Imitation Learning Methods

## Abstract

This project presents a successful reimplementation and validation of the Dataset Aggregation (DAgger) algorithm proposed by Ross et al. (2011) in the context of autonomous driving using the CarRacing-v3 environment. 

We implement and compare four distinct approaches: 
1) PPO Expert baseline
2) Behavioral Cloning (BC)
3) DAgger 
4) SMILe (Stochastic Mixing Iterative Learning)

Our results validate the core insights of the DAgger paper, demonstrating that iterative learning with expert corrections significantly outperforms traditional behavioral cloning. The improved DAgger implementation achieves near-expert performance, validating the theoretical advantages of the algorithm.

## Introduction

Imitation learning has emerged as a powerful paradigm for training autonomous agents by learning from expert demonstrations. However, traditional behavioral cloning suffers from the "distribution shift" problem, where the student policy encounters states not seen during training. The DAgger algorithm addresses this limitation through iterative data collection and retraining, allowing the student to learn from its own mistakes.

This study implements and evaluates multiple imitation learning approaches on the CarRacing-v3 environment, providing a comprehensive comparison of their effectiveness in autonomous driving scenarios.

## Methodology

### Environment Setup
- **Environment**: CarRacing-v3 (continuous action space)
- **Observation Space**: 64×64 grayscale images with 2-frame stacking
- **Action Space**: Continuous steering, acceleration, and braking
- **Evaluation Protocol**: 3 episodes per method, 1000 steps maximum per episode

### Implemented Methods

#### 1. PPO Expert (Baseline)
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Training**: 275,000 timesteps using rl-baselines3-zoo
- **Purpose**: Provides expert demonstrations for imitation learning methods

#### 2. Behavioral Cloning (BC)
- **Approach**: Single-shot imitation learning
- **Training**: 20 expert episodes, 50 epochs
- **Architecture**: CNN with 2-frame input
- **Limitation**: Suffers from distribution shift

#### 3. DAgger (Dataset Aggregation)
- **Approach**: Iterative learning with expert corrections
- **Iterations**: 20 iterations, 1 episode per iteration
- **Training**: 5 epochs per iteration with fine-tuning
- **Key Innovation**: Student collects data, expert provides corrections

#### 4. SMILe (Stochastic Mixing Iterative Learning)
- **Approach**: Ensemble of student models with stochastic mixing
- **Iterations**: 20 iterations, 1 episode per iteration
- **Architecture**: Multiple student models with α=0.1 mixing parameter
- **Advantage**: Reduces overfitting through ensemble diversity

## Results

> **Note**: All experimental results (raw logs, outputs, videos) are consolidated in [`results/baseline/`](./results/baseline/). The `check-performance.ipynb` notebook generates key metrics from these consolidated results.

### Performance Comparison

| Method | Average Reward | Std Dev | Min Reward | Max Reward |
|--------|---------------|---------|------------|------------|
| **PPO Expert** | 635.14 | 245.44 | 292.37 | 853.85 |
| **DAgger** | 625.16 | 243.62 | 319.35 | 915.50 |
| **SMILe** | 504.28 | 153.30 | 322.58 | 697.55 |
| **BC** | 51.69 | 37.90 | 22.36 | 105.21 |

### Key Findings

1. **DAgger Success**: The improved DAgger implementation achieved 98.4% of expert performance (625.16 vs 635.14), demonstrating the effectiveness of iterative learning with expert corrections and validating the core thesis of the original paper.

2. **SMILe Performance**: Achieved 79.4% of expert performance (504.28 vs 635.14), showing strong ensemble-based learning capabilities.

3. **Distribution Shift Problem**: Behavioral cloning showed poor performance (8.1% of expert), confirming the distribution shift problem that DAgger was designed to address.

4. **Iterative Learning Advantage**: Both DAgger and SMILe significantly outperformed BC, demonstrating the effectiveness of iterative learning approaches over single-shot imitation learning.


### Training Efficiency

| Method | Training Time | Total Samples | Convergence |
|--------|--------------|---------------|-------------|
| **BC** | 6.3 minutes | 18,765 | Single-shot |
| **DAgger** | 3.8 hours | 20,000 | Iterative |
| **SMILe** | 3.6 hours | 19,728 | Iterative |


## Conclusion

This reimplementation successfully validates the DAgger paper's core insights. The improved implementation demonstrates that DAgger can achieve near-expert performance (98.4% of expert reward) when properly implemented with single environment, expert mixing, and stable training procedures. The study confirms that iterative learning methods represent a significant advancement over traditional imitation learning approaches.

## Setup and Reproduction

### Prerequisites

- **Python**: 3.12 (or compatible version)
- **CUDA**: Required for GPU acceleration (recommended for expert training)

### Installation

1. **Create and activate conda environment** (recommended):
   ```bash
   conda create -n env_dagger_test python=3.12
   conda activate env_dagger_test
   ```

2. **Install system dependencies** (macOS only):
   ```bash
   brew install swig  # Required for building box2d-py
   ```

3. **Install Python packages**:
   ```bash
   pip install gymnasium stable-baselines3 tensorflow numpy rl_zoo3 "gymnasium[box2d]" "gymnasium[other]"
   ```

### Expert Training

Train the PPO expert policy:
```bash
cd expert_implementations
git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
python rl-baselines3-zoo/train.py \
    --algo ppo \
    --env CarRacing-v3 \
    --device cuda \
    --n-timesteps 275000 \
    --tensorboard-log ./ppo-carracing >> log_ppo_carracing.log 2>&1&
```

**Note**: Expert training requires GPU support and takes several hours. The trained model will be saved in the `logs/` directory.

### Imitation Learning Methods

#### Behavioral Cloning
```bash
cd behavioural_cloning
python train_bc.py
```

#### DAgger
```bash
cd dagger_implementations
python train_dagger.py
```

#### SMILe
```bash
cd SMILe_implementation
python train_smile.py
```

### Evaluation

Run the comprehensive performance comparison:
```bash
jupyter notebook check-performance.ipynb
```

The notebook automatically generates key metrics and comparison tables from consolidated experimental results.

## Experimental Results

All experimental results (logs, outputs, videos) are consolidated in the [`results/baseline/`](./results/baseline/) folder for easy access and reproducibility.

### Consolidated Results Structure
```
results/baseline/
├── expert/                    # PPO Expert results
│   ├── results_*.txt         # Performance metrics
│   ├── log_*.log             # Training logs
│   └── videos_*/             # Evaluation videos
├── behavioural_cloning/       # BC results
├── dagger/                   # DAgger results
└── SMILe/                    # SMILe results
```

### Generated Demonstrations

Video demonstrations are automatically generated during evaluation and saved in the consolidated results folder:
- **Expert**: [`results/baseline/expert/videos_20251022_012428/`](./results/baseline/expert/videos_20251022_012428/)
- **Behavioral Cloning**: [`results/baseline/behavioural_cloning/videos_20251022_155303/`](./results/baseline/behavioural_cloning/videos_20251022_155303/)
- **DAgger**: [`results/baseline/dagger/videos_20251023_074929/`](./results/baseline/dagger/videos_20251023_074929/)
- **SMILe**: [`results/baseline/SMILe/videos_20251022_161346/`](./results/baseline/SMILe/videos_20251022_161346/)

Each folder contains 3 episode recordings (MP4 format) demonstrating the agent's performance. All result files, training logs, and videos are organized in the consolidated [`results/baseline/`](./results/baseline/) directory.

## References

- Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the fourteenth international conference on artificial intelligence and statistics*.

---

*This project was completed as part of CS 8803 DRL Fall 2025 midterm assignment focusing on imitation learning algorithms.*
