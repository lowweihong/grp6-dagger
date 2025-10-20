# dagger_train_continuous.py
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from stable_baselines3 import SAC
from tqdm import tqdm
import cv2

# --- DAgger Hyperparameters ---
N_DAGGER_ITERATIONS = 20      # Total number of DAgger iterations
N_EPISODES_PER_ITERATION = 1  # Laps to collect per iteration (as per paper)
N_EPOCHS_PER_ITERATION = 5    # Keras epochs to train on the full dataset
BATCH_SIZE = 32               # Batch size for student training
EXPERT_MODEL_FILE = "./gen_expert_trajectory/sac_car_racing_expert.zip" # The trained expert
STUDENT_MODEL_FILE = "dagger_student.keras"     # Saved final student

# --- 1. Helper Function: Preprocess State ---
def preprocess_state(state):
    """Crops, grayscales, and normalizes the 96x96x3 image."""
    # Crop the bottom info panel
    state = state[0:84, :, :] 
    # Convert to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    # Add channel dimension for CNN
    state = np.expand_dims(state, axis=-1)
    # Normalize to [0, 1]
    return state.astype(np.float32) / 255.0

# --- 2. Helper Function: Build Student Model ---
def build_student_model(action_size=3):
    """Builds a CNN regression model."""
    # Using the CNN architecture from the Nature DQN paper
    model = models.Sequential([
        layers.Input(shape=(84, 84, 1)),
        layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
        layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        
        # Output layer is linear regression for [steer, gas, brake]
        # Use 'tanh' to squash outputs to [-1, 1], which matches
        # the steering range. We'll un-scale gas/brake later.
        layers.Dense(action_size, activation='tanh')
    ])
    
    # Use Mean Squared Error for regression
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mean_squared_error')
    return model

# --- 3. Helper Function: Evaluate Policy ---
def evaluate_policy(policy, env, num_episodes=5, is_student=False):
    """Runs a policy for a few episodes and returns the average reward."""
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        done = False
        episode_reward = 0
        
        while not done:
            if is_student:
                # Keras model needs batch dimension
                action = policy.predict(np.expand_dims(state, axis=0), verbose=0)[0]
                # Un-scale gas/brake (which are [0, 1]) from tanh ([-1, 1])
                # Steer [0] is fine in [-1, 1]
                action[1] = (action[1] + 1.0) / 2.0  # Gas
                action[2] = (action[2] + 1.0) / 2.0  # Brake
            else:
                # SB3 expert
                action, _ = policy.predict(state, deterministic=True)

            next_state, reward, done, _, _ = env.step(action)
            state = preprocess_state(next_state)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# --- 4. Main DAgger Training Script ---
if __name__ == '__main__':
    
    # --- Setup Environment ---
    env = gym.make("CarRacing-v3", continuous=True)
    
    # --- Load Expert ---
    print(f"Loading expert from {EXPERT_MODEL_FILE}...")
    try:
        expert = SAC.load(EXPERT_MODEL_FILE)
    except FileNotFoundError:
        print(f"Error: Expert file '{EXPERT_MODEL_FILE}' not found.")
        print("Please train your expert first and save it to this directory.")
        exit()
    
    # --- Build Student ---
    print("Building student model...")
    action_size = env.action_space.shape[0] # Should be 3
    student = build_student_model(action_size)

    # --- Initialize Dataset ---
    aggregated_states = []
    aggregated_actions = []
    
    performance_log = []

    print("--- Starting DAgger Training ---")
    for i in range(N_DAGGER_ITERATIONS):
        print(f"\n--- DAgger Iteration {i+1}/{N_DAGGER_ITERATIONS} ---")
        
        # --- Step A: Collect Data (Rollout) ---
        # On iteration 0, use the expert. After that, use the student.
        policy_to_rollout = expert if i == 0 else student
        is_expert_rollout = (i == 0)
        
        print(f"Rolling out {N_EPISODES_PER_ITERATION} episode(s) with {'EXPERT' if is_expert_rollout else 'STUDENT'} policy...")
        
        new_states_from_rollout = []
        for _ in range(N_EPISODES_PER_ITERATION):
            state, _ = env.reset()
            state = preprocess_state(state)
            done = False
            
            while not done:
                if is_expert_rollout:
                    action, _ = policy_to_rollout.predict(state, deterministic=True)
                else:
                    action = policy_to_rollout.predict(np.expand_dims(state, axis=0), verbose=0)[0]
                    # Un-scale gas/brake
                    action[1] = (action[1] + 1.0) / 2.0
                    action[2] = (action[2] + 1.0) / 2.0

                new_states_from_rollout.append(state)
                next_state, reward, done, _, _ = env.step(action)
                state = preprocess_state(next_state)

        # --- Step B: Query Expert for Labels ---
        print(f"Querying expert for {len(new_states_from_rollout)} visited states...")
        
        # SB3 predict works on batches, which is fast
        expert_actions, _ = expert.predict(
            np.array(new_states_from_rollout), 
            deterministic=True
        )
        
        # Scale expert gas/brake to [-1, 1] to match student's 'tanh' output
        # Steer [0] is already in [-1, 1]
        expert_actions_scaled = expert_actions.copy()
        expert_actions_scaled[:, 1] = (expert_actions[:, 1] * 2.0) - 1.0 # Gas
        expert_actions_scaled[:, 2] = (expert_actions[:, 2] * 2.0) - 1.0 # Brake
        
        # --- Step C: Aggregate Data ---
        aggregated_states.extend(new_states_from_rollout)
        aggregated_actions.extend(expert_actions_scaled)
        print(f"Total dataset size: {len(aggregated_states)} samples")

        # --- Step D: Retrain Student (Regression) ---
        print(f"Retraining student on aggregated dataset for {N_EPOCHS_PER_ITERATION} epochs...")
        
        student.fit(
            np.array(aggregated_states),
            np.array(aggregated_actions),
            batch_size=BATCH_SIZE,
            epochs=N_EPOCHS_PER_ITERATION,
            shuffle=True,
            verbose=1
        )

        # --- Step E: Evaluate Student Performance ---
        avg_reward = evaluate_policy(student, env, num_episodes=5, is_student=True)
        performance_log.append(avg_reward)
        print(f"--- Iteration {i+1} Student Performance: {avg_reward:.2f} ---")

    # --- 5. Final Results ---
    print("\n--- DAgger Training Complete ---")
    
    # Evaluate expert for comparison
    expert_reward = evaluate_policy(expert, env, num_episodes=10, is_student=False)
    print(f"\nFinal Expert Performance (for comparison): {expert_reward:.2f}")
    
    print("\nStudent performance per iteration (Avg. Reward):")
    for i, reward in enumerate(performance_log):
        print(f"Iteration {i+1}: {reward:.2f}")
        
    # Save the final student model
    student.save(STUDENT_MODEL_FILE)
    print(f"\nFinal student model saved to {STUDENT_MODEL_FILE}")

    env.close()