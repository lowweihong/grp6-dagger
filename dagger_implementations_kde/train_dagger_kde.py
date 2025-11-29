#!/usr/bin/env python3
"""
Modified DAgger training script with KDE-based trajectory filtering and action noise
Uses PPO as expert for CarRacing-v3
Based on the PPO model trained with rl-baselines3-zoo
Uses proper environment wrappers for preprocessing

Modifications:
1. KDE-based trajectory filtering: Only add trajectories from underexplored regions
2. Action noise: Add Gaussian noise to student policy actions with probability α
"""
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from stable_baselines3 import PPO
from tqdm import tqdm
import cv2
import os
import argparse
import json
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import collections
from collections import deque
from sklearn.neighbors import KernelDensity

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# --- DAgger Hyperparameters ---
N_DAGGER_ITERATIONS = 20      # Total number of DAgger iterations
N_EPISODES_PER_ITERATION = 1  # Episodes to collect per iteration
N_EPOCHS_PER_ITERATION = 5    # Keras epochs to train on the full dataset
BATCH_SIZE = 32               # Batch size for student training
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps per episode to prevent infinite episodes
N_STACKED_FRAMES = 2          # Number of frames to stack (must match expert)
EXPERT_MIXING_PROB = 0.1      # Probability of using expert during student rollout (β parameter)

# --- NEW: Modified DAgger Hyperparameters ---
ACTION_NOISE_PROB = 0.2       # Probability of adding noise to student policy actions (α parameter)
ACTION_NOISE_STD = 0.1        # Standard deviation of Gaussian noise (σ parameter)
KDE_BANDWIDTH = 0.5           # Bandwidth for KDE (controls smoothness)
KDE_THRESHOLD = 0.1           # Density threshold - trajectories below this are added (lower = more selective)
KDE_SAMPLE_SIZE = 1000       # Number of samples to use for KDE estimation (for efficiency)

# Default paths - can be overridden by command line arguments
DEFAULT_EXPERT_MODEL_FILE = "../expert_implementations/logs/ppo/CarRacing-v3_6/best_model.zip"
STUDENT_MODEL_FILE = "dagger_student_kde.keras"

# Manual frame stacking implementation for PPO expert
class SimpleFrameStack(gym.Wrapper):
    def __init__(self, env, n_frames=2):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = []
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize with the same frame repeated
        self.frames = [obs.copy() for _ in range(self.n_frames)]
        return np.stack(self.frames, axis=0), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add new frame and remove oldest
        self.frames.append(obs.copy())
        if len(self.frames) > self.n_frames:
            self.frames.pop(0)
        stacked_obs = np.stack(self.frames, axis=0)
        return stacked_obs, reward, terminated, truncated, info

# --- HELPER CLASS ---
class StudentFrameStack:
    """Manages frame stacking for the student policy."""
    def __init__(self, n_frames=2):
        self.n_frames = n_frames
        # Use deque for efficient L/R pop/append
        self.buffer = deque(maxlen=self.n_frames)

    def reset(self, initial_frame):
        """Resets the buffer with the initial frame."""
        processed_frame = preprocess_state_for_student(initial_frame)
        for _ in range(self.n_frames):
            self.buffer.append(processed_frame)
        # Stack along the channel axis (H, W, C)
        # The CNN expects (H, W, C) where C = n_frames
        return np.stack(self.buffer, axis=-1)

    def step(self, next_frame):
        """Adds a new frame and returns the stacked state."""
        processed_frame = preprocess_state_for_student(next_frame)
        self.buffer.append(processed_frame)
        return np.stack(self.buffer, axis=-1)

def preprocess_for_expert(state):
    """Preprocesses the 96x96x3 raw image for PPO expert query."""
    # Resize to 64x64
    state = cv2.resize(state, (64, 64), interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # PPO model expects uint8
    return state.astype(np.uint8)

def preprocess_state_for_student(state):
    """Preprocesses a single 96x96x3 image for the student stack."""
    # Resize to 64x64
    state = cv2.resize(state, (64, 64), interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # Normalize to [0, 1] - NO extra channel dimension
    return state.astype(np.float32) / 255.0

def build_student_model(action_size=3):
    """Builds a CNN regression model for the student."""
    # Using the CNN architecture from the Nature DQN paper
    
    # Input shape is now (64, 64, N_STACKED_FRAMES)
    inputs = layers.Input(shape=(64, 64, N_STACKED_FRAMES))
    
    # CNN layers
    x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
    x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    
    # Split the output head and apply the correct activations
    # This matches the PPO expert's output scaling
    steer_out = layers.Dense(1, activation='tanh', name='steer')(x)    # Range [-1, 1]
    gas_out = layers.Dense(1, activation='sigmoid', name='gas')(x)      # Range [0, 1]
    brake_out = layers.Dense(1, activation='sigmoid', name='brake')(x)  # Range [0, 1]
    
    # Concatenate them back into a single [steer, gas, brake] tensor
    outputs = layers.Concatenate(name='actions')([steer_out, gas_out, brake_out])
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use Mean Squared Error for regression with lower learning rate for stability
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='mean_squared_error')
    return model

def add_action_noise(action, action_space, noise_std):
    """
    Add Gaussian noise to action and clip to valid bounds.
    
    Args:
        action: Original action [steer, gas, brake]
        action_space: Gym action space with low/high bounds
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Noisy action clipped to valid bounds
    """
    noise = np.random.normal(0, noise_std, size=action.shape)
    noisy_action = action + noise
    
    # Clip to action space bounds
    noisy_action = np.clip(noisy_action, action_space.low, action_space.high)
    return noisy_action

def compute_state_features(state):
    """
    Extract features from state for KDE computation.
    Since states are images (64, 64, N_STACKED_FRAMES), we use summary statistics
    to reduce dimensionality and improve numerical stability.
    
    Args:
        state: State array of shape (64, 64, N_STACKED_FRAMES)
    
    Returns:
        Feature vector with summary statistics
    """
    # Use summary statistics instead of raw pixels for better numerical stability
    # This reduces dimensionality and makes KDE more stable
    features = []
    
    # For each frame in the stack
    for frame_idx in range(state.shape[-1]):
        frame = state[:, :, frame_idx]
        # Add mean, std, and some quantiles
        features.extend([
            np.mean(frame),
            np.std(frame),
            np.percentile(frame, 25),
            np.percentile(frame, 50),
            np.percentile(frame, 75),
        ])
    
    # Add some spatial features (center vs edges)
    center_region = state[24:40, 24:40, :].mean()
    edge_region = np.concatenate([
        state[:16, :, :].mean(axis=(0, 1)),
        state[-16:, :, :].mean(axis=(0, 1)),
        state[:, :16, :].mean(axis=(0, 1)),
        state[:, -16:, :].mean(axis=(0, 1))
    ]).mean()
    features.extend([center_region, edge_region])
    
    return np.array(features)

def filter_trajectories_with_kde(new_states, existing_states, kde_threshold, kde_bandwidth, kde_sample_size):
    """
    Filter trajectories using KDE to only keep those from underexplored regions.
    
    Args:
        new_states: List of new state arrays to potentially add
        existing_states: List of existing state arrays in dataset
        kde_threshold: Density threshold - keep states below this (in log space)
        kde_bandwidth: Bandwidth for KDE
        kde_sample_size: Max number of samples to use for KDE (for efficiency)
    
    Returns:
        filtered_indices: Indices of new_states that should be kept (low density)
        densities: Density values for all new states (for display)
        log_densities: Log density values (for numerical stability and debugging)
    """
    if len(existing_states) == 0:
        # If no existing data, keep all new states
        # Return very negative log densities to indicate unexplored
        log_densities = np.full(len(new_states), -np.inf)
        densities = np.zeros(len(new_states))
        return list(range(len(new_states))), densities, log_densities
    
    # Convert states to feature vectors
    new_features = np.array([compute_state_features(state) for state in new_states])
    existing_features = np.array([compute_state_features(state) for state in existing_states])
    
    # Normalize features for better numerical stability
    feature_mean = existing_features.mean(axis=0)
    feature_std = existing_features.std(axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    existing_features_norm = (existing_features - feature_mean) / feature_std
    new_features_norm = (new_features - feature_mean) / feature_std
    
    # Subsample existing states for KDE if too many (for efficiency)
    if len(existing_features_norm) > kde_sample_size:
        indices = np.random.choice(len(existing_features_norm), kde_sample_size, replace=False)
        existing_features_norm = existing_features_norm[indices]
    
    # Fit KDE on existing dataset
    kde = KernelDensity(bandwidth=kde_bandwidth, kernel='gaussian')
    kde.fit(existing_features_norm)
    
    # Evaluate log density for new states (work in log space for numerical stability)
    log_densities = kde.score_samples(new_features_norm)
    
    # Convert threshold to log space for comparison
    # If threshold is in density space, convert: log_threshold = log(threshold)
    # But we'll work directly with log densities and use a log threshold
    log_threshold = np.log(kde_threshold) if kde_threshold > 0 else -np.inf
    
    # Keep states with log density below log threshold (underexplored regions)
    filtered_indices = np.where(log_densities < log_threshold)[0].tolist()
    
    # Return densities in original space for display (with clipping to avoid underflow)
    # Clip very negative log densities to avoid exp() underflow
    log_densities_clipped = np.clip(log_densities, -50, None)  # exp(-50) ≈ 0, but still representable
    densities = np.exp(log_densities_clipped)
    
    return filtered_indices, densities, log_densities

def evaluate_policy(policy, env, num_episodes=5, is_student=False):
    """Runs a policy for a few episodes and returns the average reward and fall count."""
    total_rewards = []
    total_falls = []
    
    for episode in range(num_episodes):
        
        # Need to handle student vs expert env state differently
        if is_student:
            # Student env returns raw 96x96x3 frames
            state_raw, _ = env.reset()
            # Use same frame stacking as main training loop
            student_frame_stack = collections.deque(maxlen=N_STACKED_FRAMES)
            student_processed = preprocess_state_for_student(state_raw)
            for _ in range(N_STACKED_FRAMES):
                student_frame_stack.append(student_processed)
            state = np.stack(student_frame_stack, axis=-1)
        else:
            # Expert env is wrapped, state is already 64x64x2 (stacked by axis 0)
            state, _ = env.reset()

        done = False
        episode_reward = 0
        step_count = 0
        fall_count = 0
        
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            if is_student:
                action = policy.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            else:
                # For PPO expert: state is already preprocessed by wrappers
                action, _ = policy.predict(state, deterministic=True)

            # The env.step() returns the *next* state
            if is_student:
                next_state_raw, reward, terminated, truncated, _ = env.step(action)
                # Update frame stack with new frame
                student_frame_stack.append(preprocess_state_for_student(next_state_raw))
                next_state = np.stack(student_frame_stack, axis=-1)
            else:
                next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Track falls - CarRacing-v3 gives negative reward when off track
            if reward < -10:  # Typical fall reward is around -100
                fall_count += 1
        
        total_rewards.append(episode_reward)
        total_falls.append(fall_count)
    
    avg_reward = np.mean(total_rewards)
    avg_falls = np.mean(total_falls)
    return avg_reward, avg_falls

def main():
    parser = argparse.ArgumentParser(description='Train Modified DAgger with KDE filtering and action noise')
    parser.add_argument('--expert-model', type=str, default=DEFAULT_EXPERT_MODEL_FILE,
                        help='Path to the trained PPO expert model')
    parser.add_argument('--student-model', type=str, default=STUDENT_MODEL_FILE,
                        help='Path to save the student model')
    parser.add_argument('--iterations', type=int, default=N_DAGGER_ITERATIONS,
                        help='Number of DAgger iterations')
    parser.add_argument('--episodes-per-iter', type=int, default=N_EPISODES_PER_ITERATION,
                        help='Number of episodes per iteration')
    parser.add_argument('--epochs-per-iter', type=int, default=N_EPOCHS_PER_ITERATION,
                        help='Number of epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for student training')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of episodes for evaluation')
    parser.add_argument('--tensorboard-log', type=str, default='./dagger_tensorboard_kde',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    
    # New hyperparameters
    parser.add_argument('--expert-mixing-prob', type=float, default=EXPERT_MIXING_PROB,
                        help='Probability of using expert policy (β parameter)')
    parser.add_argument('--action-noise-prob', type=float, default=ACTION_NOISE_PROB,
                        help='Probability of adding noise to student actions (α parameter)')
    parser.add_argument('--action-noise-std', type=float, default=ACTION_NOISE_STD,
                        help='Standard deviation of action noise (σ parameter)')
    parser.add_argument('--kde-threshold', type=float, default=KDE_THRESHOLD,
                        help='KDE density threshold - keep trajectories below this')
    parser.add_argument('--kde-bandwidth', type=float, default=KDE_BANDWIDTH,
                        help='Bandwidth for KDE')
    parser.add_argument('--kde-sample-size', type=int, default=KDE_SAMPLE_SIZE,
                        help='Max samples to use for KDE estimation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Modified DAgger Training with KDE Filtering and Action Noise")
    print("=" * 60)
    print(f"Expert model: {args.expert_model}")
    print(f"Student model: {args.student_model}")
    print(f"Iterations: {args.iterations}")
    print(f"Episodes per iteration: {args.episodes_per_iter}")
    print(f"Epochs per iteration: {args.epochs_per_iter}")
    print(f"Batch size: {args.batch_size}")
    print(f"Frame stacking: {N_STACKED_FRAMES} frames")
    print(f"\n--- Modified DAgger Parameters ---")
    print(f"Expert mixing probability (β): {args.expert_mixing_prob}")
    print(f"Action noise probability (α): {args.action_noise_prob}")
    print(f"Action noise std (σ): {args.action_noise_std}")
    print(f"KDE threshold: {args.kde_threshold}")
    print(f"KDE bandwidth: {args.kde_bandwidth}")
    print(f"KDE sample size: {args.kde_sample_size}")
    print(f"TensorBoard logging: {'Disabled' if args.no_tensorboard else f'Enabled ({args.tensorboard_log})'}")
    print("=" * 60)
    
    # --- Setup TensorBoard ---
    tensorboard_callback = None
    if not args.no_tensorboard:
        # Create TensorBoard callback
        tensorboard_callback = TensorBoard(
            log_dir=args.tensorboard_log,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0
        )
        print(f"TensorBoard logs will be saved to: {args.tensorboard_log}")
        print(f"To view: tensorboard --logdir {args.tensorboard_log}")
    
    # --- Setup Environment (Single Environment Like SMILe) ---
    print("Creating single rollout environment...")
    # Use ONE raw environment for rollouts to prevent divergence
    env = gym.make("CarRacing-v3", continuous=True)
    
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")
    
    # --- Load PPO Expert ---
    print(f"Loading PPO expert from {args.expert_model}...")
    try:
        expert = PPO.load(args.expert_model)
        print("PPO expert loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Expert file '{args.expert_model}' not found.")
        print("Please train your PPO expert first using:")
        print("python rl-baselines3-zoo/train.py --algo ppo --env CarRacing-v3 --device cuda --n-timesteps 400000 --tensorboard-log ./ppo-carracing")
        return
    except Exception as e:
        print(f"Error loading PPO expert: {e}")
        return
    
    # --- Build Student ---
    print("Building student model...")
    action_size = env.action_space.shape[0]  # Should be 3
    student = build_student_model(action_size)
    print(f"Student model built with action size: {action_size}, input shape (64, 64, {N_STACKED_FRAMES})")

    # --- Initialize Dataset ---
    aggregated_states = []
    aggregated_actions = []
    performance_log = []
    fall_log = []  # Track falls per iteration
    filtering_stats = []  # Track KDE filtering statistics
    
    # Custom metrics for TensorBoard
    if not args.no_tensorboard:
        import tensorflow as tf
        # Create a summary writer for custom metrics
        summary_writer = tf.summary.create_file_writer(args.tensorboard_log)

    print("\n--- Starting Modified DAgger Training ---")
    start_time = datetime.now()
    
    for i in range(args.iterations):
        print(f"\n--- DAgger Iteration {i+1}/{args.iterations} ---")
        
        # --- Step A: Collect Data (Rollout with Expert Mixing and Action Noise) ---
        print(f"Rolling out {args.episodes_per_iter} episode(s) with Modified DAgger...")
        
        new_states_student_format = []
        new_states_expert_format = []
        iteration_falls = 0
        
        for episode_idx in range(args.episodes_per_iter):
            # Reset env and frame stacks
            raw_state, _ = env.reset()
            
            expert_frame_stack = collections.deque(maxlen=N_STACKED_FRAMES)
            student_frame_stack = collections.deque(maxlen=N_STACKED_FRAMES)

            # Initialize stacks
            expert_processed = preprocess_for_expert(raw_state)
            student_processed = preprocess_state_for_student(raw_state)
            for _ in range(N_STACKED_FRAMES):
                expert_frame_stack.append(expert_processed)
            for _ in range(N_STACKED_FRAMES):
                student_frame_stack.append(student_processed)

            done = False
            step_count = 0
            episode_reward = 0
            episode_falls = 0
            
            while not done and step_count < MAX_STEPS_PER_EPISODE:
                # Create stacked states from deques
                # Expert state: (N_STACKED_FRAMES, 64, 64)
                state_expert = np.stack(expert_frame_stack, axis=0)
                # Student state: (64, 64, N_STACKED_FRAMES) - stack along channel axis
                state_student = np.stack(student_frame_stack, axis=-1)

                # --- MODIFIED: Action Selection with Expert Mixing and Noise ---
                rand_val = np.random.rand()
                
                if i == 0 or rand_val < args.expert_mixing_prob:
                    # Use expert policy (iteration 0 or with mixing probability β)
                    policy_action, _ = expert.predict(state_expert, deterministic=True)
                else:
                    # Use student policy
                    policy_action = student.predict(np.expand_dims(state_student, axis=0), verbose=0)[0]
                    
                    # --- NEW: Add action noise with probability α ---
                    if np.random.rand() < args.action_noise_prob:
                        policy_action = add_action_noise(
                            policy_action, 
                            env.action_space, 
                            args.action_noise_std
                        )
                
                # --- Store state *before* stepping ---
                # We store the state we were in, to be labeled by the expert
                new_states_student_format.append(state_student)
                new_states_expert_format.append(state_expert)
                
                # --- Step the single environment ---
                next_raw_state, reward, terminated, truncated, _ = env.step(policy_action)
                done = terminated or truncated
                
                # Count falls (reward -10 indicates falling off track)
                if reward == -10:
                    episode_falls += 1
                
                episode_reward += reward
                step_count += 1
                
                # Update stacks with the new state
                expert_frame_stack.append(preprocess_for_expert(next_raw_state))
                student_frame_stack.append(preprocess_state_for_student(next_raw_state))
            
            iteration_falls += episode_falls
            print(f"  Episode {episode_idx+1}: {step_count} steps, reward: {episode_reward:.2f}, falls: {episode_falls}")

        # --- Step B: Query Expert for Labels ---
        print(f"Querying expert for {len(new_states_expert_format)} visited states...")
        
        # Use expert format states for querying the expert
        expert_query_states = np.array(new_states_expert_format)
        
        # PPO predict works on batches, which is fast
        expert_actions, _ = expert.predict(
            expert_query_states, 
            deterministic=True
        )
        # expert_actions are already in the correct [-1,1], [0,1] format
        
        # --- Step C: MODIFIED - Filter Trajectories with KDE before Aggregating ---
        print(f"Filtering trajectories using KDE (threshold={args.kde_threshold})...")
        
        filtered_indices, densities, log_densities = filter_trajectories_with_kde(
            new_states_student_format,
            aggregated_states,
            args.kde_threshold,
            args.kde_bandwidth,
            args.kde_sample_size
        )
        
        num_filtered = len(new_states_student_format) - len(filtered_indices)
        filtering_stats.append({
            'total_sampled': len(new_states_student_format),
            'filtered_out': num_filtered,
            'kept': len(filtered_indices),
            'filter_ratio': num_filtered / len(new_states_student_format) if len(new_states_student_format) > 0 else 0.0,
            'avg_density': float(np.mean(densities)),
            'min_density': float(np.min(densities)),
            'max_density': float(np.max(densities)),
            'avg_log_density': float(np.mean(log_densities)),
            'min_log_density': float(np.min(log_densities)),
            'max_log_density': float(np.max(log_densities))
        })
        
        print(f"  Sampled: {len(new_states_student_format)}, Filtered out: {num_filtered}, Kept: {len(filtered_indices)}")
        print(f"  Density stats (exp space) - Mean: {np.mean(densities):.6e}, Min: {np.min(densities):.6e}, Max: {np.max(densities):.6e}")
        print(f"  Log density stats - Mean: {np.mean(log_densities):.2f}, Min: {np.min(log_densities):.2f}, Max: {np.max(log_densities):.2f}")
        
        # Only add filtered trajectories
        filtered_states = [new_states_student_format[idx] for idx in filtered_indices]
        filtered_actions = [expert_actions[idx] for idx in filtered_indices]
        
        aggregated_states.extend(filtered_states)
        aggregated_actions.extend(filtered_actions)
        print(f"Total dataset size: {len(aggregated_states)} samples (added {len(filtered_states)} new samples)")

        # --- Step D: Fine-tune Student (Like SMILe) ---
        print(f"Fine-tuning student on aggregated dataset for {args.epochs_per_iter} epochs...")

        # Debug: Check data quality
        states_array = np.array(aggregated_states)
        actions_array = np.array(aggregated_actions)
        print(f"Dataset stats - States shape: {states_array.shape}, Actions shape: {actions_array.shape}")
        print(f"States range: [{states_array.min():.3f}, {states_array.max():.3f}]")
        print(f"Actions range - Steer: [{actions_array[:, 0].min():.3f}, {actions_array[:, 0].max():.3f}], "
              f"Gas: [{actions_array[:, 1].min():.3f}, {actions_array[:, 1].max():.3f}], "
              f"Brake: [{actions_array[:, 2].min():.3f}, {actions_array[:, 2].max():.3f}]")
        
        
        # Prepare callbacks
        callbacks = []
        if tensorboard_callback:
            # Update TensorBoard log directory for this iteration
            tensorboard_callback.log_dir = f"{args.tensorboard_log}/iteration_{i+1}"
            callbacks.append(tensorboard_callback)
        
        # Add checkpoint callback to save model after each iteration
        os.makedirs("results", exist_ok=True)
        checkpoint_path = os.path.join("results", f"checkpoint_iter_{i+1}.keras")
        checkpoint_callback = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=False,
            verbose=0
        )
        callbacks.append(checkpoint_callback)
        
        # Fine-tune the student model (like SMILe) instead of retraining from scratch
        student.fit(
            states_array,
            actions_array,
            batch_size=args.batch_size,
            epochs=args.epochs_per_iter,
            shuffle=True,
            verbose=1,
            callbacks=callbacks,
            validation_split=0.1,
        )

        # --- Step E: Evaluate Student Performance ---
        avg_reward, avg_falls = evaluate_policy(student, env, num_episodes=args.eval_episodes, is_student=True)
        performance_log.append(avg_reward)
        fall_log.append(avg_falls)
        print(f"--- Iteration {i+1} Student Performance: Reward={avg_reward:.2f}, Avg Falls={avg_falls:.1f} ---")
        print(f"--- Iteration {i+1} Data Collection Falls: {iteration_falls} ---")

        # Debug: Test student action prediction on a few states
        if i < 3:  # Only for first few iterations
            test_states = states_array[:5]  # Test on first 5 states
            test_actions = actions_array[:5]
            student_predictions = student.predict(test_states)
            print(f"Sample predictions vs targets (Model output is already scaled):")
            for j in range(3):
                print(f"  State {j}: Pred=[{student_predictions[j][0]:.3f}, {student_predictions[j][1]:.3f}, {student_predictions[j][2]:.3f}] "
                      f"Target=[{test_actions[j][0]:.3f}, {test_actions[j][1]:.3f}, {test_actions[j][2]:.3f}]")
        
        
        # Log custom metrics to TensorBoard
        if not args.no_tensorboard:
            with summary_writer.as_default():
                tf.summary.scalar('Student_Performance/Reward', avg_reward, step=i+1)
                tf.summary.scalar('Student_Performance/Avg_Falls', avg_falls, step=i+1)
                tf.summary.scalar('Data_Collection/Falls', iteration_falls, step=i+1)
                tf.summary.scalar('Dataset/Total_Samples', len(aggregated_states), step=i+1)
                tf.summary.scalar('Dataset/New_Samples_Added', len(filtered_states), step=i+1)
                tf.summary.scalar('KDE_Filtering/Filter_Ratio', filtering_stats[-1]['filter_ratio'], step=i+1)
                tf.summary.scalar('KDE_Filtering/Avg_Density', filtering_stats[-1]['avg_density'], step=i+1)
                tf.summary.scalar('KDE_Filtering/Sampled', filtering_stats[-1]['total_sampled'], step=i+1)
                tf.summary.scalar('KDE_Filtering/Kept', filtering_stats[-1]['kept'], step=i+1)
            summary_writer.flush()

    # --- Final Results ---
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("Modified DAgger Training Complete!")
    print("=" * 60)
    
    # Evaluate expert for comparison
    # Create proper expert environment with wrappers
    expert_env = gym.make("CarRacing-v3", continuous=True)
    from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
    expert_env = ResizeObservation(expert_env, shape=(64, 64))
    expert_env = GrayscaleObservation(expert_env, keep_dim=False)
    expert_env = SimpleFrameStack(expert_env, n_frames=N_STACKED_FRAMES)
    
    expert_reward, expert_falls = evaluate_policy(expert, expert_env, num_episodes=10, is_student=False)
    print(f"Final Expert Performance (for comparison): Reward={expert_reward:.2f}, Avg Falls={expert_falls:.1f}")
    
    print("\nStudent performance per iteration:")
    print("Iteration | Reward | Avg Falls | Filter Ratio | Avg Density")
    print("----------|--------|----------|-------------|-------------")
    for i, (reward, falls, stats) in enumerate(zip(performance_log, fall_log, filtering_stats)):
        print(f"{i+1:9d} | {reward:6.2f} | {falls:8.1f} | {stats['filter_ratio']:11.2%} | {stats['avg_density']:12.4f}")
    
    # Calculate improvement
    if len(performance_log) > 1:
        reward_improvement = performance_log[-1] - performance_log[0]
        fall_improvement = fall_log[0] - fall_log[-1]  # Lower falls is better
        print(f"\nTotal improvement:")
        print(f"  Reward: {reward_improvement:.2f}")
        print(f"  Falls: {fall_improvement:.1f} (lower is better)")
        print(f"Final performance vs Expert:")
        print(f"  Reward: {performance_log[-1]:.2f} vs {expert_reward:.2f}")
        print(f"  Falls: {fall_log[-1]:.1f} vs {expert_falls:.1f}")
    
    print(f"\nTraining time: {training_time}")
    print(f"Total samples collected: {len(aggregated_states)}")
    
    # Save the final student model
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    # If student_model path already includes results/, use it as-is, otherwise add results/
    if args.student_model.startswith("results/"):
        model_path = args.student_model
    else:
        model_path = os.path.join("results", args.student_model)
    student.save(model_path)
    print(f"\nFinal student model saved to {model_path}")
    
    # Save performance log
    log_file = f"dagger_kde_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        f.write(f"Modified DAgger (KDE + Action Noise) Training Results\n")
        f.write(f"Expert Performance: Reward={expert_reward:.2f}, Avg Falls={expert_falls:.1f}\n")
        f.write(f"Training Time: {training_time}\n")
        f.write(f"Total Samples: {len(aggregated_states)}\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  Expert mixing prob (β): {args.expert_mixing_prob}\n")
        f.write(f"  Action noise prob (α): {args.action_noise_prob}\n")
        f.write(f"  Action noise std (σ): {args.action_noise_std}\n")
        f.write(f"  KDE threshold: {args.kde_threshold}\n")
        f.write(f"  KDE bandwidth: {args.kde_bandwidth}\n")
        f.write("\nStudent Performance per Iteration:\n")
        f.write("Iteration | Reward | Avg Falls | Filter Ratio | Avg Density\n")
        f.write("----------|--------|----------|-------------|-------------\n")
        for i, (reward, falls, stats) in enumerate(zip(performance_log, fall_log, filtering_stats)):
            f.write(f"{i+1:9d} | {reward:6.2f} | {falls:8.1f} | {stats['filter_ratio']:11.2%} | {stats['avg_density']:12.4f}\n")
    
    print(f"Performance log saved to {log_file}")
    
    # Save JSON results
    json_file = f"dagger_kde_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_data = {
        "training_info": {
            "expert_model": args.expert_model,
            "student_model": args.student_model,
            "iterations": args.iterations,
            "episodes_per_iteration": args.episodes_per_iter,
            "epochs_per_iteration": args.epochs_per_iter,
            "batch_size": args.batch_size,
            "eval_episodes": args.eval_episodes,
            "training_time_seconds": training_time.total_seconds(),
            "total_samples": len(aggregated_states),
            "n_stacked_frames": N_STACKED_FRAMES,
            "expert_mixing_prob": args.expert_mixing_prob,
            "action_noise_prob": args.action_noise_prob,
            "action_noise_std": args.action_noise_std,
            "kde_threshold": args.kde_threshold,
            "kde_bandwidth": args.kde_bandwidth,
            "kde_sample_size": args.kde_sample_size
        },
        "expert_performance": {
            "reward": float(expert_reward),
            "avg_falls": float(expert_falls)
        },
        "student_performance": {
            "iterations": []
        },
        "kde_filtering_stats": {
            "iterations": []
        },
        "improvements": {}
    }
    
    # Add iteration data
    for i, (reward, falls, stats) in enumerate(zip(performance_log, fall_log, filtering_stats)):
        results_data["student_performance"]["iterations"].append({
            "iteration": i + 1,
            "reward": float(reward),
            "avg_falls": float(falls)
        })
        results_data["kde_filtering_stats"]["iterations"].append({
            "iteration": i + 1,
            "total_sampled": stats["total_sampled"],
            "filtered_out": stats["filtered_out"],
            "kept": stats["kept"],
            "filter_ratio": float(stats["filter_ratio"]),
            "avg_density": stats["avg_density"],
            "min_density": stats["min_density"],
            "max_density": stats["max_density"],
            "avg_log_density": stats["avg_log_density"],
            "min_log_density": stats["min_log_density"],
            "max_log_density": stats["max_log_density"]
        })
    
    # Add improvement data
    if len(performance_log) > 1:
        results_data["improvements"] = {
            "reward_improvement": float(performance_log[-1] - performance_log[0]),
            "fall_improvement": float(fall_log[0] - fall_log[-1]),  # Lower falls is better
            "final_vs_expert": {
                "reward": {
                    "student": float(performance_log[-1]),
                    "expert": float(expert_reward),
                    "difference": float(performance_log[-1] - expert_reward)
                },
                "falls": {
                    "student": float(fall_log[-1]),
                    "expert": float(expert_falls),
                    "difference": float(fall_log[-1] - expert_falls)
                }
            }
        }
    
    with open(json_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"JSON results saved to {json_file}")
    
    # Final TensorBoard logging
    if not args.no_tensorboard:
        with summary_writer.as_default():
            tf.summary.scalar('Final/Expert_Reward', expert_reward, step=0)
            tf.summary.scalar('Final/Expert_Falls', expert_falls, step=0)
            tf.summary.scalar('Final/Student_Reward', performance_log[-1], step=0)
            tf.summary.scalar('Final/Student_Falls', fall_log[-1], step=0)
            tf.summary.scalar('Final/Training_Time_Hours', training_time.total_seconds() / 3600, step=0)
        summary_writer.flush()
        summary_writer.close()
        print(f"\nTensorBoard logs saved to: {args.tensorboard_log}")
        print(f"To view: tensorboard --logdir {args.tensorboard_log}")
    
    env.close()
    expert_env.close()

if __name__ == '__main__':
    main()

