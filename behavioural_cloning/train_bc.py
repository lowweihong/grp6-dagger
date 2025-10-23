#!/usr/bin/env python3
"""
Behavioral Cloning (BC) training script using PPO as expert for CarRacing-v3
Based on the DAgger script, but modified for a single, static dataset.
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
from tensorflow.keras.callbacks import TensorBoard
from collections import deque # Added for frame stacking

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# --- BC Hyperparameters ---
N_COLLECTION_EPISODES = 20    # Total expert episodes to collect for the dataset
N_TRAINING_EPOCHS = 50        # Keras epochs to train on the full dataset
BATCH_SIZE = 32               # Batch size for student training
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps per episode
N_STACKED_FRAMES = 2          # Number of frames to stack (must match expert)

# Default paths - can be overridden by command line arguments
DEFAULT_EXPERT_MODEL_FILE = "../expert_implementations/logs/ppo/CarRacing-v3_6/best_model.zip"
DEFAULT_STUDENT_MODEL_FILE = "results/bc_student_model.keras" # Changed name
DEFAULT_TENSORBOARD_LOG = "./results/bc_tensorboard"         # Changed name

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
        return np.concatenate(self.buffer, axis=-1)

    def step(self, next_frame):
        """Adds a new frame and returns the stacked state."""
        processed_frame = preprocess_state_for_student(next_frame)
        self.buffer.append(processed_frame)
        return np.concatenate(self.buffer, axis=-1)
# --- END HELPER CLASS ---

def create_expert_env():
    """Create environment with proper wrappers for PPO expert."""
    env = gym.make("CarRacing-v3", continuous=True)
    
    # Apply the same preprocessing as rl-baselines3-zoo
    from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
    env = ResizeObservation(env, shape=(64, 64))
    env = GrayscaleObservation(env, keep_dim=False)
    env = SimpleFrameStack(env, n_frames=N_STACKED_FRAMES)
    
    return env

def create_student_env():
    """Create environment for student data collection (returns raw frames)."""
    return gym.make("CarRacing-v3", continuous=True)

def preprocess_state_for_student(state):
    """Preprocesses a single 96x96x3 image for the student stack."""
    # Resize to 64x64
    state = cv2.resize(state, (64, 64), interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # Add channel dimension for CNN
    state = np.expand_dims(state, axis=-1)
    # Normalize to [0, 1]
    return state.astype(np.float32) / 255.0

def build_student_model(action_size=3):
    """Builds a CNN regression model for the student."""
    # Input shape is (64, 64, N_STACKED_FRAMES)
    inputs = layers.Input(shape=(64, 64, N_STACKED_FRAMES))
    
    # CNN layers
    x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
    x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    
    # Split the output head and apply the correct activations
    steer_out = layers.Dense(1, activation='tanh', name='steer')(x)      # Range [-1, 1]
    gas_out = layers.Dense(1, activation='sigmoid', name='gas')(x)      # Range [0, 1]
    brake_out = layers.Dense(1, activation='sigmoid', name='brake')(x)  # Range [0, 1]
    
    # Concatenate them back into a single [steer, gas, brake] tensor
    outputs = layers.Concatenate(name='actions')([steer_out, gas_out, brake_out])
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mean_squared_error')
    return model

def evaluate_policy(policy, env, num_episodes=5, is_student=False):
    """Runs a policy for a few episodes and returns the average reward and fall count."""
    total_rewards = []
    total_falls = []
    
    for episode in range(num_episodes):
        
        if is_student:
            # Student env returns raw 96x96x3 frames
            state_raw, _ = env.reset()
            student_stacker = StudentFrameStack(n_frames=N_STACKED_FRAMES)
            state = student_stacker.reset(state_raw)
        else:
            # Expert env is wrapped, state is already 64x64x2 (stacked by axis 0)
            state, _ = env.reset()

        done = False
        episode_reward = 0
        step_count = 0
        fall_count = 0
        
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            if is_student:
                # Keras model needs batch dimension
                action = policy.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            else:
                # For PPO expert
                action, _ = policy.predict(state, deterministic=True)

            if is_student:
                next_state_raw, reward, terminated, truncated, _ = env.step(action)
                next_state = student_stacker.step(next_state_raw)
            else:
                next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Track falls
            if reward < -10: 
                fall_count += 1
        
        total_rewards.append(episode_reward)
        total_falls.append(fall_count)
    
    avg_reward = np.mean(total_rewards)
    avg_falls = np.mean(total_falls)
    return avg_reward, avg_falls

def main():
    parser = argparse.ArgumentParser(description='Train Behavioral Cloning (BC) on CarRacing-v3')
    parser.add_argument('--expert-model', type=str, default=DEFAULT_EXPERT_MODEL_FILE,
                        help='Path to the trained PPO expert model')
    parser.add_argument('--student-model', type=str, default=DEFAULT_STUDENT_MODEL_FILE,
                        help='Path to save the student model')
    # --- MODIFIED ARGS ---
    parser.add_argument('--collection-episodes', type=int, default=N_COLLECTION_EPISODES,
                        help='Number of expert episodes to collect for the dataset')
    parser.add_argument('--training-epochs', type=int, default=N_TRAINING_EPOCHS,
                        help='Number of epochs to train the student')
    # --- END MODIFIED ARGS ---
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for student training')
    parser.add_argument('--eval-episodes', type=int, default=10, # Increased default for final eval
                        help='Number of episodes for evaluation')
    parser.add_argument('--tensorboard-log', type=str, default=DEFAULT_TENSORBOARD_LOG,
                        help='Directory for TensorBoard logs')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Behavioral Cloning (BC) Training on CarRacing-v3")
    print("=" * 60)
    print(f"Expert model: {args.expert_model}")
    print(f"Student model: {args.student_model}")
    print(f"Collection Episodes: {args.collection_episodes}")
    print(f"Training Epochs: {args.training_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Frame stacking: {N_STACKED_FRAMES} frames")
    print(f"TensorBoard logging: {'Disabled' if args.no_tensorboard else f'Enabled ({args.tensorboard_log})'}")
    print("=" * 60)
    
    # --- Setup TensorBoard ---
    tensorboard_callback = None
    if not args.no_tensorboard:
        tensorboard_callback = TensorBoard(
            log_dir=args.tensorboard_log,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        print(f"TensorBoard logs will be saved to: {args.tensorboard_log}")
        print(f"To view: tensorboard --logdir {args.tensorboard_log}")
    
    # --- Setup Environments ---
    print("Creating environments...")
    expert_env = create_expert_env()
    student_env = create_student_env()
    
    # --- Load PPO Expert ---
    print(f"Loading PPO expert from {args.expert_model}...")
    try:
        expert = PPO.load(args.expert_model)
        print("PPO expert loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Expert file '{args.expert_model}' not found.")
        return
    except Exception as e:
        print(f"Error loading PPO expert: {e}")
        return
    
    # --- Build Student ---
    print("Building student model...")
    action_size = student_env.action_space.shape[0] # Should be 3
    student = build_student_model(action_size)
    print(f"Student model built with action size: {action_size}, input shape (64, 64, {N_STACKED_FRAMES})")

    # --- Initialize Dataset ---
    aggregated_states = []
    aggregated_actions = []
    
    print("\n--- Starting BC Data Collection ---")
    start_time = datetime.now()
    
    # --- Step 1: Collect Static Dataset ---
    print(f"Collecting data from {args.collection_episodes} expert episodes...")
    
    total_collection_falls = 0
    
    for episode_idx in tqdm(range(args.collection_episodes), desc="Collecting Episodes"):
        # Reset both environments
        state_expert, _ = expert_env.reset()
        state_student_raw, _ = student_env.reset()
        
        # Initialize the student's frame stacker
        student_stacker = StudentFrameStack(n_frames=N_STACKED_FRAMES)
        state_student = student_stacker.reset(state_student_raw)
        
        done = False
        step_count = 0
        episode_falls = 0
        
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            # Get action from expert
            action, _ = expert.predict(state_expert, deterministic=True)
            
            # Step both environments with the *same* expert action
            next_state_expert, reward_expert, terminated_expert, truncated_expert, _ = expert_env.step(action)
            next_state_student_raw, reward_student, terminated_student, truncated_student, _ = student_env.step(action)
            
            # Get the *next* stacked state for the student
            next_state_student = student_stacker.step(next_state_student_raw)
            
            reward = reward_student
            done = terminated_student or truncated_student
            
            step_count += 1
            
            if reward < -10:
                episode_falls += 1
            
            # --- Store the data ---
            # We store the student's view of the state (s_t)
            # and the expert's action (a_t)
            aggregated_states.append(state_student)
            aggregated_actions.append(action)
            
            # Update to next state
            state_expert = next_state_expert
            state_student = next_state_student # This is now the stacked (s_{t+1})
        
        total_collection_falls += episode_falls

    print(f"Data collection complete. Total samples: {len(aggregated_states)}")
    print(f"Total falls during collection: {total_collection_falls}")
    
    # --- Step 2: Train Student on Aggregated Data ---
    print(f"\nTraining student on aggregated dataset for {args.training_epochs} epochs...")

    states_array = np.array(aggregated_states)
    actions_array = np.array(aggregated_actions)
    
    print(f"Dataset stats - States shape: {states_array.shape}, Actions shape: {actions_array.shape}")
    
    callbacks = []
    if tensorboard_callback:
        callbacks.append(tensorboard_callback)
    
    student.fit(
        states_array,
        actions_array,
        batch_size=args.batch_size,
        epochs=args.training_epochs,
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("Student training complete!")

    # --- Step 3: Evaluate Final Policies ---
    print("\n" + "=" * 60)
    print("Evaluating Final Policies")
    print("=" * 60)
    
    # Evaluate expert
    print(f"Evaluating expert for {args.eval_episodes} episodes...")
    expert_reward, expert_falls = evaluate_policy(expert, expert_env, num_episodes=args.eval_episodes, is_student=False)
    print(f"--- Expert Performance: Reward={expert_reward:.2f}, Avg Falls={expert_falls:.1f} ---")
    
    # Evaluate student
    print(f"Evaluating BC student for {args.eval_episodes} episodes...")
    student_reward, student_falls = evaluate_policy(student, student_env, num_episodes=args.eval_episodes, is_student=True)
    print(f"--- BC Student Performance: Reward={student_reward:.2f}, Avg Falls={student_falls:.1f} ---")

    
    # --- Final Results ---
    print("\n" + "=" * 60)
    print("BC Training Complete!")
    print("=" * 60)
    print(f"Training time: {training_time}")
    print(f"Total samples collected: {len(aggregated_states)}")
    print(f"Final Expert Performance:   Reward={expert_reward:.2f}, Avg Falls={expert_falls:.1f}")
    print(f"Final Student Performance: Reward={student_reward:.2f}, Avg Falls={student_falls:.1f}")
    
    # Save the final student model
    student.save(args.student_model)
    print(f"\nFinal student model saved to {args.student_model}")
    
    # --- Save Logs ---
    log_file = f"bc_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        f.write(f"BC Training Results\n")
        f.write(f"Training Time: {training_time}\n")
        f.write(f"Total Samples: {len(aggregated_states)}\n\n")
        f.write(f"Expert Performance: Reward={expert_reward:.2f}, Avg Falls={expert_falls:.1f}\n")
        f.write(f"Student Performance: Reward={student_reward:.2f}, Avg Falls={student_falls:.1f}\n")
    print(f"Performance log saved to {log_file}")
    
    json_file = f"bc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_data = {
        "training_info": {
            "type": "Behavioral Cloning",
            "expert_model": args.expert_model,
            "student_model": args.student_model,
            "collection_episodes": args.collection_episodes,
            "training_epochs": args.training_epochs,
            "batch_size": args.batch_size,
            "eval_episodes": args.eval_episodes,
            "training_time_seconds": training_time.total_seconds(),
            "total_samples": len(aggregated_states),
            "n_stacked_frames": N_STACKED_FRAMES
        },
        "expert_performance": {
            "reward": float(expert_reward),
            "avg_falls": float(expert_falls)
        },
        "student_performance": {
            "reward": float(student_reward),
            "avg_falls": float(student_falls)
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"JSON results saved to {json_file}")
    
    # Final TensorBoard logging
    if not args.no_tensorboard:
        try:
            summary_writer = tf.summary.create_file_writer(args.tensorboard_log)
            with summary_writer.as_default():
                tf.summary.scalar('Final/Expert_Reward', expert_reward, step=0)
                tf.summary.scalar('Final/Expert_Falls', expert_falls, step=0)
                tf.summary.scalar('Final/Student_Reward', student_reward, step=0)
                tf.summary.scalar('Final/Student_Falls', student_falls, step=0)
                tf.summary.scalar('Final/Training_Time_Hours', training_time.total_seconds() / 3600, step=0)
                tf.summary.scalar('Final/Total_Samples', len(aggregated_states), step=0)
            summary_writer.flush()
            summary_writer.close()
            print(f"\nFinal metrics saved to TensorBoard: {args.tensorboard_log}")
        except Exception as e:
            print(f"Error writing final TensorBoard summaries: {e}")
    
    expert_env.close()
    student_env.close()

if __name__ == '__main__':
    main()
# nohup python train_bc.py >> log_train_bc.log 2>&1&