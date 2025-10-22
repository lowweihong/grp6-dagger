#!/usr/bin/env python3
"""
DAgger training script using PPO as expert for CarRacing-v3
Based on the PPO model trained with rl-baselines3-zoo
Uses proper environment wrappers for preprocessing
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
from datetime import datetime

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# --- DAgger Hyperparameters ---
N_DAGGER_ITERATIONS = 20      # Total number of DAgger iterations
N_EPISODES_PER_ITERATION = 1  # Episodes to collect per iteration
N_EPOCHS_PER_ITERATION = 5    # Keras epochs to train on the full dataset
BATCH_SIZE = 32               # Batch size for student training
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps per episode to prevent infinite episodes

# Default paths - can be overridden by command line arguments
DEFAULT_EXPERT_MODEL_FILE = "./logs/ppo/CarRacing-v3_6/best_model.zip"
STUDENT_MODEL_FILE = "dagger_student_ppo2.keras"

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

def create_expert_env():
    """Create environment with proper wrappers for PPO expert."""
    env = gym.make("CarRacing-v3", continuous=True)
    
    # Apply the same preprocessing as rl-baselines3-zoo
    from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
    env = ResizeObservation(env, shape=(64, 64))
    env = GrayscaleObservation(env, keep_dim=False)
    env = SimpleFrameStack(env, n_frames=2)
    
    return env

def create_student_env():
    """Create environment for student data collection."""
    return gym.make("CarRacing-v3", continuous=True)

def preprocess_state_for_student(state):
    """Preprocesses the 96x96x3 image for student model."""
    # For student: resize to 64x64 and convert to grayscale (same as expert)
    # Resize to 64x64
    state = cv2.resize(state, (64, 64), interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # Add channel dimension for CNN
    state = np.expand_dims(state, axis=-1)
    # Normalize to [0, 1]
    return state.astype(np.float32) / 255.0

def postprocess_student_action(action):
    """Convert raw student model output to correct action ranges."""
    # Raw output: [steer_raw, gas_raw, brake_raw]
    # Target ranges: steer=[-1,1], gas=[0,1], brake=[0,1]
    
    # Use tanh for steering to get [-1, 1]
    steer = np.tanh(action[0])
    
    # Use sigmoid for gas and brake to get [0, 1]
    gas = 1.0 / (1.0 + np.exp(-action[1]))  # sigmoid
    brake = 1.0 / (1.0 + np.exp(-action[2]))  # sigmoid
    
    return np.array([steer, gas, brake])

def build_student_model(action_size=3):
    """Builds a CNN regression model for the student."""
    # Using the CNN architecture from the Nature DQN paper
    # Input shape matches expert preprocessing: (64, 64, 1)
    inputs = layers.Input(shape=(64, 64, 1))
    
    # CNN layers
    x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
    x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    
    # Single output layer - outputs raw values
    outputs = layers.Dense(3, activation='linear', name='actions')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use Mean Squared Error for regression
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mean_squared_error')
    return model

def evaluate_policy(policy, env, num_episodes=5, is_student=False):
    """Runs a policy for a few episodes and returns the average reward."""
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            if is_student:
                # For student: preprocess the raw state
                state_processed = preprocess_state_for_student(state)
                # Keras model needs batch dimension
                action_raw = policy.predict(np.expand_dims(state_processed, axis=0), verbose=0)[0]
                # Convert raw output to correct action ranges
                action = postprocess_student_action(action_raw)
            else:
                # For PPO expert: state is already preprocessed by wrappers
                action, _ = policy.predict(state, deterministic=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            step_count += 1
        
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def main():
    parser = argparse.ArgumentParser(description='Train DAgger with PPO expert on CarRacing-v3')
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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DAgger Training with PPO Expert on CarRacing-v3")
    print("=" * 60)
    print(f"Expert model: {args.expert_model}")
    print(f"Student model: {args.student_model}")
    print(f"Iterations: {args.iterations}")
    print(f"Episodes per iteration: {args.episodes_per_iter}")
    print(f"Epochs per iteration: {args.epochs_per_iter}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # --- Setup Environments ---
    print("Creating environments...")
    expert_env = create_expert_env()
    student_env = create_student_env()
    
    print(f"Expert env observation space: {expert_env.observation_space}")
    print(f"Student env observation space: {student_env.observation_space}")
    
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
    action_size = student_env.action_space.shape[0]  # Should be 3
    student = build_student_model(action_size)
    print(f"Student model built with action size: {action_size}")

    # --- Initialize Dataset ---
    aggregated_states = []
    aggregated_actions = []
    performance_log = []

    print("\n--- Starting DAgger Training ---")
    start_time = datetime.now()
    
    for i in range(args.iterations):
        print(f"\n--- DAgger Iteration {i+1}/{args.iterations} ---")
        
        # --- Step A: Collect Data (Rollout) ---
        # On iteration 0, use the expert. After that, use the student.
        policy_to_rollout = expert if i == 0 else student
        is_expert_rollout = (i == 0)
        
        print(f"Rolling out {args.episodes_per_iter} episode(s) with {'EXPERT' if is_expert_rollout else 'STUDENT'} policy...")
        
        new_states_from_rollout = []
        new_states_expert_format = []  # Store states in expert format for querying
        
        for episode_idx in range(args.episodes_per_iter):
            # Reset both environments
            state_expert, _ = expert_env.reset()
            state_student_raw, _ = student_env.reset()
            state_student = preprocess_state_for_student(state_student_raw)
            
            done = False
            step_count = 0
            episode_reward = 0
            
            while not done and step_count < MAX_STEPS_PER_EPISODE:
                if is_expert_rollout:
                    # Use expert policy
                    action, _ = policy_to_rollout.predict(state_expert, deterministic=True)
                else:
                    # Use student policy
                    action_raw = policy_to_rollout.predict(np.expand_dims(state_student, axis=0), verbose=0)[0]
                    # Convert raw output to correct action ranges
                    action = postprocess_student_action(action_raw)
                
                # Step both environments with the same action
                next_state_expert, reward_expert, terminated_expert, truncated_expert, _ = expert_env.step(action)
                next_state_student_raw, reward_student, terminated_student, truncated_student, _ = student_env.step(action)
                next_state_student = preprocess_state_for_student(next_state_student_raw)
                
                # Use student environment reward and termination
                reward = reward_student
                terminated = terminated_student
                truncated = truncated_student
                done = terminated or truncated
                
                episode_reward += reward
                step_count += 1
                
                # Collect the state we just visited (before updating to next state)
                new_states_from_rollout.append(state_student)
                new_states_expert_format.append(state_expert)
                
                # Update to next state
                state_expert = next_state_expert
                state_student = next_state_student
            
            print(f"  Episode {episode_idx+1}: {step_count} steps, reward: {episode_reward:.2f}")

        # --- Step B: Query Expert for Labels ---
        print(f"Querying expert for {len(new_states_from_rollout)} visited states...")
        
        # Use expert format states for querying the expert
        expert_query_states = np.array(new_states_expert_format)
        
        # PPO predict works on batches, which is fast
        expert_actions, _ = expert.predict(
            expert_query_states, 
            deterministic=True
        )
        
        # Expert actions are already in the correct format for the student model
        # Steer: [-1, 1], Gas: [0, 1], Brake: [0, 1]
        expert_actions_scaled = expert_actions.copy()
        
        # --- Step C: Aggregate Data ---
        aggregated_states.extend(new_states_from_rollout)
        aggregated_actions.extend(expert_actions_scaled)
        print(f"Total dataset size: {len(aggregated_states)} samples")

        # --- Step D: Retrain Student (Regression) ---
        print(f"Retraining student on aggregated dataset for {args.epochs_per_iter} epochs...")
        
        student.fit(
            np.array(aggregated_states),
            np.array(aggregated_actions),
            batch_size=args.batch_size,
            epochs=args.epochs_per_iter,
            shuffle=True,
            verbose=1
        )

        # --- Step E: Evaluate Student Performance ---
        avg_reward = evaluate_policy(student, student_env, num_episodes=args.eval_episodes, is_student=True)
        performance_log.append(avg_reward)
        print(f"--- Iteration {i+1} Student Performance: {avg_reward:.2f} ---")

    # --- Final Results ---
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("DAgger Training Complete!")
    print("=" * 60)
    
    # Evaluate expert for comparison
    expert_reward = evaluate_policy(expert, expert_env, num_episodes=10, is_student=False)
    print(f"Final Expert Performance (for comparison): {expert_reward:.2f}")
    
    print("\nStudent performance per iteration (Avg. Reward):")
    for i, reward in enumerate(performance_log):
        print(f"Iteration {i+1}: {reward:.2f}")
    
    # Calculate improvement
    if len(performance_log) > 1:
        improvement = performance_log[-1] - performance_log[0]
        print(f"\nTotal improvement: {improvement:.2f}")
        print(f"Final performance vs Expert: {performance_log[-1]:.2f} vs {expert_reward:.2f}")
    
    print(f"\nTraining time: {training_time}")
    print(f"Total samples collected: {len(aggregated_states)}")
    
    # Save the final student model
    student.save(args.student_model)
    print(f"\nFinal student model saved to {args.student_model}")
    
    # Save performance log
    log_file = f"dagger_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        f.write(f"DAgger Training Results\n")
        f.write(f"Expert Performance: {expert_reward:.2f}\n")
        f.write(f"Training Time: {training_time}\n")
        f.write(f"Total Samples: {len(aggregated_states)}\n\n")
        f.write("Student Performance per Iteration:\n")
        for i, reward in enumerate(performance_log):
            f.write(f"Iteration {i+1}: {reward:.2f}\n")
    
    print(f"Performance log saved to {log_file}")
    expert_env.close()
    student_env.close()

if __name__ == '__main__':
    main()

# nohup python train_dagger_2.py >> log_dagger2.log 2>&1&
