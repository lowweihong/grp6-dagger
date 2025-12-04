#!/usr/bin/env python3
"""
SMILe training script using PPO as expert for CarRacing-v3
SMILe (Stochastic Mixing Iterative Learning)

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
import collections
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import json

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- SMILe Hyperparameters ---
N_ITERATIONS = 20        # Total number of iterations
N_EPISODES_PER_ITERATION = 1    # Episodes to collect per iteration
N_EPOCHS_PER_ITERATION = 5      # Keras epochs to train on the full dataset
BATCH_SIZE = 32                 # Batch size for student training
MAX_STEPS_PER_EPISODE = 1000    # Maximum steps per episode
ALPHA = 0.1                     # SMILe mixing parameter (probability of using expert)
N_EXPERT_FRAMES = 2             # Frame stack size for PPO expert
N_STUDENT_FRAMES = 2            # Frame stack size for student (must match expert logic)

# Default paths - can be overridden by command line arguments
DEFAULT_EXPERT_MODEL_FILE = "../expert_implementations/logs/ppo/CarRacing-v3_6/best_model.zip"
STUDENT_MODEL_FILE = "results/student_smile.keras"

# --- Manual frame stacking wrapper for PPO expert EVALUATION ---
# This is ONLY used for the final evaluation of the expert
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

def create_expert_eval_env():
    """Create environment with proper wrappers for PPO expert EVALUATION."""
    env = gym.make("CarRacing-v3", continuous=True)
    
    # Apply the same preprocessing as rl-baselines3-zoo
    from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
    env = ResizeObservation(env, shape=(64, 64))
    env = GrayscaleObservation(env, keep_dim=False)
    env = SimpleFrameStack(env, n_frames=N_EXPERT_FRAMES)
    
    return env

def preprocess_for_expert(state):
    """Preprocesses the 96x96x3 raw image for PPO expert query."""
    # Resize to 64x64
    state = cv2.resize(state, (64, 64), interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # PPO model expects uint8
    return state.astype(np.uint8)

def preprocess_state_for_student(state):
    """Preprocesses the 96x96x3 raw image for student model."""
    # Resize to 64x64
    state = cv2.resize(state, (64, 64), interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # Normalize to [0, 1]. Returns (64, 64)
    return state.astype(np.float32) / 255.0

def build_student_model(input_shape, action_size=3):
    """
    Builds a CNN regression model for the student.
    Input shape should be (64, 64, N_STUDENT_FRAMES)
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers (Nature DQN architecture)
    x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
    x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    
    # --- Corrected Output Layers ---
    # Use activations to directly output actions in the correct range
    steer_output = layers.Dense(1, activation='tanh', name='steer')(x)   # Range [-1, 1]
    gas_output = layers.Dense(1, activation='sigmoid', name='gas')(x)   # Range [0, 1]
    brake_output = layers.Dense(1, activation='sigmoid', name='brake')(x) # Range [0, 1]
    
    # Concatenate the outputs
    outputs = layers.Concatenate(name='actions')([steer_output, gas_output, brake_output])
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use Mean Squared Error for regression
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mean_squared_error')
    return model

def evaluate_student_policy(policy, num_episodes=5):
    """Runs the student policy and returns the average reward and fall count."""
    # Use a raw env and manage frame stack manually
    env = gym.make("CarRacing-v3", continuous=True)
    total_rewards = []
    total_falls = []
    
    for _ in range(num_episodes):
        raw_state, _ = env.reset()
        
        # Initialize student frame stack
        student_frame_stack = collections.deque(maxlen=N_STUDENT_FRAMES)
        processed = preprocess_state_for_student(raw_state)
        for _ in range(N_STUDENT_FRAMES):
            student_frame_stack.append(processed)
            
        done = False
        episode_reward = 0
        episode_falls = 0
        step_count = 0
        
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            # Create stack: (64, 64, N_STUDENT_FRAMES)
            state_student = np.stack(student_frame_stack, axis=-1)
            
            # Get action (model output is already in correct range)
            action = policy.predict(np.expand_dims(state_student, axis=0), verbose=0)[0]

            next_raw_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Count falls (reward -10 indicates falling off track)
            if reward == -10:
                episode_falls += 1
            
            # Add new preprocessed frame to stack
            student_frame_stack.append(preprocess_state_for_student(next_raw_state))
            
            episode_reward += reward
            step_count += 1
            
        total_rewards.append(episode_reward)
        total_falls.append(episode_falls)
    env.close()
    return np.mean(total_rewards), np.mean(total_falls)

def evaluate_expert_policy(policy, num_episodes=5):
    """Runs the PPO expert policy and returns the average reward and fall count."""
    # Use the fully wrapped env
    env = create_expert_eval_env()
    total_rewards = []
    total_falls = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_falls = 0
        step_count = 0
        
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            # For PPO expert: state is already preprocessed by wrappers
            action, _ = policy.predict(state, deterministic=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Count falls (reward -10 indicates falling off track)
            if reward == -10:
                episode_falls += 1
                
            state = next_state
            episode_reward += reward
            step_count += 1
            
        total_rewards.append(episode_reward)
        total_falls.append(episode_falls)
    env.close()
    return np.mean(total_rewards), np.mean(total_falls)

def compute_smile_mixing_probabilities(iteration, alpha=ALPHA):
    """
    Compute SMILe mixing probabilities for current iteration.
    """
    # Probability of using expert decreases over iterations
    expert_prob = (1 - alpha) ** iteration
    
    # Weights for student models (if any exist)
    if iteration > 0:
        # Probabilities for student models 1 to iteration
        student_probs = [alpha * (1 - alpha) ** (j - 1) for j in range(1, iteration + 1)]
        # Total probability mass for students
        total_student_prob = sum(student_probs)
        
        if total_student_prob > 0:
            # Normalize to get weights
            student_weights = [p / total_student_prob for p in student_probs]
        else:
            student_weights = []
    else:
        student_weights = []
        
    return expert_prob, student_weights

def main():
    parser = argparse.ArgumentParser(description='Train SMILe with PPO expert on CarRacing-v3')
    parser.add_argument('--expert-model', type=str, default=DEFAULT_EXPERT_MODEL_FILE, help='Path to the trained PPO expert model')
    parser.add_argument('--student-model', type=str, default=STUDENT_MODEL_FILE, help='Path to save the student model')
    parser.add_argument('--iterations', type=int, default=N_ITERATIONS, help='Number of iterations')
    parser.add_argument('--episodes-per-iter', type=int, default=N_EPISODES_PER_ITERATION, help='Number of episodes per iteration')
    parser.add_argument('--epochs-per-iter', type=int, default=N_EPOCHS_PER_ITERATION, help='Number of epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for student training')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='SMILe mixing parameter')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of episodes for evaluation')
    parser.add_argument('--tensorboard-logdir', type=str, default='./tensorboard_logs_smile', help='Directory for TensorBoard logs')
    parser.add_argument('--results-file', type=str, default=None, help='JSON file to save detailed results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Corrected SMILe Training on CarRacing-v3")
    print("=" * 60)
    print(f"Expert model: {args.expert_model}")
    print(f"Student model: {args.student_model}")
    print(f"Iterations: {args.iterations}, Episodes/Iter: {args.episodes_per_iter}")
    print(f"Epochs/Iter: {args.epochs_per_iter}, Batch Size: {args.batch_size}")
    print(f"SMILe alpha: {args.alpha}, Expert Frames: {N_EXPERT_FRAMES}, Student Frames: {N_STUDENT_FRAMES}")
    print("=" * 60)
    
    # --- Setup Environments ---
    print("Creating single rollout environment...")
    # Use ONE raw environment for rollouts to prevent divergence
    env = gym.make("CarRacing-v3", continuous=True)
    
    # --- Load PPO Expert ---
    print(f"Loading PPO expert from {args.expert_model}...")
    try:
        expert = PPO.load(args.expert_model)
        print("PPO expert loaded successfully!")
    except Exception as e:
        print(f"Error loading PPO expert: {e}")
        env.close()
        return
    
    # --- Build Student ---
    print("Building student model...")
    action_size = env.action_space.shape[0] # Should be 3
    student_input_shape = (64, 64, N_STUDENT_FRAMES)
    
    # Build ONE student model that will be fine-tuned
    student = build_student_model(student_input_shape, action_size)
    print(f"Student model built with input shape {student_input_shape}")

    # --- Initialize Dataset ---
    aggregated_states = []
    aggregated_actions = []
    performance_log = []
    fall_count_log = []
    # Store all *past* student models for SMILe mixing
    # These are separate instances
    student_models = []
    
    # --- Setup TensorBoard ---
    os.makedirs(args.tensorboard_logdir, exist_ok=True)
    tensorboard_callback = TensorBoard(
        log_dir=args.tensorboard_logdir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    ) 

    print("\n--- Starting SMILe Training ---")
    start_time = datetime.now()
    
    for i in range(args.iterations):
        print(f"\n--- SMILe Iteration {i+1}/{args.iterations} ---")
        
        # --- Compute SMILe Mixing Probabilities ---
        expert_prob, student_weights = compute_smile_mixing_probabilities(i, args.alpha)
        # Note: student_weights list has length 'i'. student_models list also has length 'i'.
        print(f"SMILe mixing - Expert prob: {expert_prob:.4f}")
        if student_weights:
             print(f"Student weights (for {len(student_weights)} models): {[round(w, 3) for w in student_weights]}")
        
        # --- Step A: Collect Data (Rollout with Stochastic Mixing) ---
        print(f"Rolling out {args.episodes_per_iter} episode(s) with SMILe stochastic mixing...")
        
        new_states_student_format = []
        new_states_expert_format = []
        iteration_falls = 0
        
        for episode_idx in range(args.episodes_per_iter):
            # Reset env and frame stacks
            raw_state, _ = env.reset()
            
            expert_frame_stack = collections.deque(maxlen=N_EXPERT_FRAMES)
            student_frame_stack = collections.deque(maxlen=N_STUDENT_FRAMES)

            # Initialize stacks
            expert_processed = preprocess_for_expert(raw_state)
            student_processed = preprocess_state_for_student(raw_state)
            for _ in range(N_EXPERT_FRAMES):
                expert_frame_stack.append(expert_processed)
            for _ in range(N_STUDENT_FRAMES):
                student_frame_stack.append(student_processed)

            done = False
            step_count = 0
            episode_reward = 0
            episode_falls = 0
            
            while not done and step_count < MAX_STEPS_PER_EPISODE:
                # Create stacked states from deques
                # Expert state: (N_EXPERT_FRAMES, 64, 64)
                state_expert = np.stack(expert_frame_stack, axis=0)
                # Student state: (64, 64, N_STUDENT_FRAMES)
                state_student = np.stack(student_frame_stack, axis=-1)

                # --- SMILe Stochastic Mixing: decide which policy to use ---
                if np.random.rand() < expert_prob:
                    # Use expert policy
                    policy_action, _ = expert.predict(state_expert, deterministic=True)
                else:
                    # Use student policy (or mixture of student models)
                    if not student_models:
                        # Should not happen after iter 0, but as a fallback
                        policy_action, _ = expert.predict(state_expert, deterministic=True)
                    else:
                        # Sample from student models based on weights
                        model_idx = np.random.choice(len(student_models), p=student_weights)
                        
                        # Get action (model output is already in correct range)
                        policy_action = student_models[model_idx].predict(
                            np.expand_dims(state_student, axis=0), verbose=0
                        )[0]
                
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
        
        # --- Step C: Aggregate Data ---
        aggregated_states.extend(new_states_student_format)
        aggregated_actions.extend(expert_actions)
        print(f"Total dataset size: {len(aggregated_states)} samples")

        # --- Step D: Train Student Model ---
        print(f"Fine-tuning student model on aggregated dataset for {args.epochs_per_iter} epochs...")
        
        # Fine-tune the *main* student model (warm-starting)
        history = student.fit(
            np.array(aggregated_states),
            np.array(aggregated_actions),
            batch_size=args.batch_size,
            epochs=args.epochs_per_iter,
            shuffle=True,
            verbose=1,
            callbacks=[tensorboard_callback]
        )
        
        # --- Save a *copy* of this model for the SMILe mixture ---
        temp_weights_file = f"temp_model_iter_{i}.weights.h5"
        student.save_weights(temp_weights_file)
        
        # Create a new, separate model instance for the mixture list
        model_for_mixing = build_student_model(student_input_shape, action_size)
        model_for_mixing.load_weights(temp_weights_file)
        student_models.append(model_for_mixing)
        
        os.remove(temp_weights_file) # Clean up temp file
        
        # --- Step E: Evaluate *latest* Student Performance ---
        avg_reward, avg_falls = evaluate_student_policy(student, num_episodes=args.eval_episodes)
        performance_log.append(avg_reward)
        fall_count_log.append(avg_falls)
        
        # Log to TensorBoard
        with tf.summary.create_file_writer(args.tensorboard_logdir).as_default():
            tf.summary.scalar('Student/Average_Reward', avg_reward, step=i)
            tf.summary.scalar('Student/Average_Falls', avg_falls, step=i)
            tf.summary.scalar('Data_Collection/Falls_Per_Iteration', iteration_falls, step=i)
            tf.summary.scalar('Data_Collection/Dataset_Size', len(aggregated_states), step=i)
            tf.summary.scalar('SMILe/Expert_Probability', expert_prob, step=i)
            if history and 'loss' in history.history:
                tf.summary.scalar('Training/Final_Loss', history.history['loss'][-1], step=i)
        
        print(f"--- Iteration {i+1} Student Performance: {avg_reward:.2f}, Falls: {avg_falls:.2f} ---")

    # --- Final Results ---
    end_time = datetime.now()
    training_time = end_time - start_time
    env.close()
    
    print("\n" + "=" * 60)
    print("SMILe Training Complete!")
    print("=" * 60)
    
    # Evaluate expert for comparison
    print("Evaluating final expert performance...")
    expert_reward, expert_falls = evaluate_expert_policy(expert, num_episodes=10)
    print(f"Final Expert Performance (for comparison): {expert_reward:.2f}, Falls: {expert_falls:.2f}")
    
    print("\nStudent performance per iteration (Avg. Reward, Avg. Falls):")
    for i, (reward, falls) in enumerate(zip(performance_log, fall_count_log)):
        print(f"Iteration {i+1}: Reward: {reward:.2f}, Falls: {falls:.2f}")
    
    print(f"\nTraining time: {training_time}")
    print(f"Total samples collected: {len(aggregated_states)}")
    print(f"Number of student models in mixture: {len(student_models)}")
    
    # Save the final, fine-tuned student model
    student.save(args.student_model)
    print(f"\nFinal student model saved to {args.student_model}")
    
    # Save performance log with fall counts
    log_file = f"smile_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        f.write(f"SMILe Training Results (with Fall Tracking)\n")
        f.write(f"Expert Performance: Reward: {expert_reward:.2f}, Falls: {expert_falls:.2f}\n")
        f.write(f"Training time: {training_time}\n")
        f.write(f"Total samples collected: {len(aggregated_states)}\n")
        f.write(f"Number of student models in mixture: {len(student_models)}\n")
        f.write(f"TensorBoard logs saved to: {args.tensorboard_logdir}\n\n")
        f.write("Student performance per iteration:\n")
        for i, (reward, falls) in enumerate(zip(performance_log, fall_count_log)):
            f.write(f"Iteration {i+1}: Reward: {reward:.2f}, Falls: {falls:.2f}\n")
    print(f"Performance log saved to {log_file}")
    print(f"TensorBoard logs saved to {args.tensorboard_logdir}")
    print("To view TensorBoard logs, run: tensorboard --logdir=" + args.tensorboard_logdir)
    
    # --- Save Detailed JSON Results ---
    if args.results_file is None:
        json_file = f"smile_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        json_file = args.results_file
    
    results_data = {
        "experiment_type": "SMILe",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "iterations": args.iterations,
            "episodes_per_iteration": args.episodes_per_iter,
            "epochs_per_iteration": args.epochs_per_iter,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "expert_frames": N_EXPERT_FRAMES,
            "student_frames": N_STUDENT_FRAMES,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE
        },
        "expert_performance": {
            "reward": float(expert_reward),
            "falls": float(expert_falls)
        },
        "training_time_seconds": float(training_time.total_seconds()),
        "total_samples_collected": len(aggregated_states),
        "number_of_student_models": len(student_models),
        "iterations": []
    }
    
    # Add per-iteration data
    for i, (reward, falls) in enumerate(zip(performance_log, fall_count_log)):
        results_data["iterations"].append({
            "iteration": i + 1,
            "reward": float(reward),
            "avg_falls": float(falls)
        })
    
    # Add improvement data
    if len(performance_log) > 1:
        results_data["improvements"] = {
            "reward_improvement": float(performance_log[-1] - performance_log[0]),
            "fall_improvement": float(fall_count_log[0] - fall_count_log[-1]),  # Lower falls is better
            "final_vs_expert": {
                "reward": {
                    "student": float(performance_log[-1]),
                    "expert": float(expert_reward),
                    "difference": float(performance_log[-1] - expert_reward)
                },
                "falls": {
                    "student": float(fall_count_log[-1]),
                    "expert": float(expert_falls),
                    "difference": float(fall_count_log[-1] - expert_falls)
                }
            }
        }
    
    with open(json_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"JSON results saved to {json_file}")
    
    # Final TensorBoard logging
    with tf.summary.create_file_writer(args.tensorboard_logdir).as_default():
        tf.summary.scalar('Final/Expert_Reward', expert_reward, step=0)
        tf.summary.scalar('Final/Expert_Falls', expert_falls, step=0)
        tf.summary.scalar('Final/Student_Reward', performance_log[-1], step=0)
        tf.summary.scalar('Final/Student_Falls', fall_count_log[-1], step=0)
        tf.summary.scalar('Final/Training_Time_Hours', training_time.total_seconds() / 3600, step=0)
    
    print("\n" + "=" * 60)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 60)
    print(f"Expert Performance: Reward: {expert_reward:.2f}, Falls: {expert_falls:.2f}")
    print(f"Final Student Performance: Reward: {performance_log[-1]:.2f}, Falls: {fall_count_log[-1]:.2f}")
    
    if len(performance_log) > 1:
        reward_improvement = performance_log[-1] - performance_log[0]
        fall_improvement = fall_count_log[0] - fall_count_log[-1]
        print(f"Reward Improvement: {reward_improvement:.2f}")
        print(f"Fall Improvement: {fall_improvement:.2f} (lower is better)")
        
        reward_vs_expert = performance_log[-1] - expert_reward
        fall_vs_expert = fall_count_log[-1] - expert_falls
        print(f"Final vs Expert - Reward: {reward_vs_expert:.2f}")
        print(f"Final vs Expert - Falls: {fall_vs_expert:.2f}")
    
    print(f"Training Time: {training_time}")
    print(f"Total Samples: {len(aggregated_states)}")
    print(f"Student Models in Mixture: {len(student_models)}")
    print("=" * 60)

if __name__ == '__main__':
    main()

# nohup python train_smile.py >> results/log_smile.log 2>&1&
