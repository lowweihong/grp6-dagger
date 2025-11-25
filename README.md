# DAgger Paper Reimplementation: A Comprehensive Study of Imitation Learning Methods

## Abstract

This project presents a successful reimplementation and validation of the Dataset Aggregation (DAgger) algorithm proposed by Ross et al. (2011) in the context of autonomous driving using the CarRacing-v3 environment. 

We implement and compare four distinct approaches: 
1) PPO Expert baseline
2) Behavioral Cloning (BC)
3) DAgger 
4) SMILe (Stochastic Mixing Iterative Learning)
5) Modified DAgger (with KDE filtering and action noise)

Our results validate the core insights of the DAgger paper, demonstrating that iterative learning with expert corrections significantly outperforms traditional behavioral cloning. The improved DAgger implementation achieves near-expert performance, validating the theoretical advantages of the algorithm.

## Introduction

Imitation learning has emerged as a powerful paradigm for training autonomous agents by learning from expert demonstrations. However, traditional behavioral cloning suffers from the "distribution shift" problem, where the student policy encounters states not seen during training. The DAgger algorithm addresses this limitation through iterative data collection and retraining, allowing the student to learn from its own mistakes.

This study implements and evaluates multiple imitation learning approaches on the CarRacing-v3 environment, providing a comprehensive comparison of their effectiveness in autonomous driving scenarios.

## Methodology

### Environment Setup
- **Environment**: CarRacing-v3 (continuous action space)
- **Observation**: 64x64 grayscale images with 2-frame stacking
- **Action Space**: Continuous steering, acceleration, and braking
- **Evaluation**: 3 episodes per method, 1000 steps maximum per episode

### Implemented Methods

#### 1. PPO Expert (Baseline)
- **Algorithm**: Proximal Policy Optimization
- **Training**: 275,000 timesteps using rl-baselines3-zoo
- **Purpose**: Provides expert demonstrations for imitation learning

#### 2. Behavioral Cloning (BC)
- **Approach**: Single-shot imitation learning
- **Training**: 20 expert episodes, 50 epochs
- **Architecture**: CNN with 2-frame input
- **Limitation**: Suffers from distribution shift

#### 3. DAgger (Dataset Aggregation)
- **Approach**: Iterative learning with expert corrections
- **Iterations**: 20 iterations, 1 episode per iteration
- **Training**: 5 epochs per iteration
- **Key Innovation**: Student collects data, expert provides corrections

#### 4. SMILe (Stochastic Mixing Iterative Learning)
- **Approach**: Ensemble of student models with stochastic mixing
- **Iterations**: 20 iterations, 1 episode per iteration
- **Architecture**: Multiple student models with α=0.1 mixing parameter
- **Advantage**: Reduces overfitting through ensemble diversity

## Results

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

## Discussion

### DAgger Success Factors
- **Single Environment**: Eliminated environment divergence issues
- **Expert Mixing**: 10% expert mixing during rollout prevents getting stuck in bad states
- **Fine-tuning**: Stable learning through iterative fine-tuning instead of retraining
- **Lower Learning Rate**: Improved convergence with 1e-5 learning rate

### SMILe Success Factors
- **Ensemble Diversity**: Multiple models reduce overfitting
- **Stochastic Mixing**: α=0.1 parameter balances exploration and exploitation
- **Iterative Learning**: Addresses distribution shift effectively


## Conclusion

This reimplementation successfully validates the DAgger paper's core insights. The improved implementation demonstrates that DAgger can achieve near-expert performance (98.4% of expert reward) when properly implemented with single environment, expert mixing, and stable training procedures. The study confirms that iterative learning methods represent a significant advancement over traditional imitation learning approaches.

## Setup and Reproduction

### Prerequisites

Install all required dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

Alternatively, if you prefer to install dependencies manually:

```bash
pip install gymnasium stable-baselines3 tensorflow numpy tqdm opencv-python scikit-learn jupyter notebook matplotlib pillow
```

**Note**: For the Modified DAgger implementation, `scikit-learn` is required for KDE-based trajectory filtering.

### Expert Training
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

### Imitation Learning Methods
```bash
# Behavioral Cloning
cd behavioural_cloning
python train_bc.py

# DAgger
cd dagger_implementations
python train_dagger_with_fallcount.py

# SMILe
cd SMILe_implementation
python train_smile.py

# Modified DAgger (with KDE filtering and action noise)
cd dagger_modified
python train_dagger_modified.py \
    --expert-model ../expert_implementations/logs/ppo/CarRacing-v3_6/best_model.zip \
    --kde-threshold 0.5 \
    --action-noise-prob 0.1 \
    --action-noise-std 0.1
```

### Evaluation
```bash
# Run comprehensive comparison
jupyter notebook check-performance.ipynb
```

### Generated Demonstrations

Video demonstrations are automatically generated during evaluation and saved in timestamped folders:
- **Expert**: [`expert_implementations/videos_20251022_012428/`](./expert_implementations/videos_20251022_012428/)
- **Behavioral Cloning**: [`behavioural_cloning/bc_videos_20251022_155303/`](./behavioural_cloning/bc_videos_20251022_155303/)
- **DAgger**: [`dagger_implementations/dagger_videos_20251023_074929/`](./dagger_implementations/dagger_videos_20251023_074929/)
- **SMILe**: [`SMILe_implementation/smile_videos_20251022_161346/`](./SMILe_implementation/smile_videos_20251022_161346/)

Each folder contains 3 episode recordings (MP4 format) demonstrating the agent's performance.

## References

- Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the fourteenth international conference on artificial intelligence and statistics*.

---

*This project was completed as part of CS 8803 DRL Fall 2025 midterm assignment focusing on imitation learning algorithms.*
