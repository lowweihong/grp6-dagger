# train_sac_expert.py
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Create the continuous environment
env = gym.make("CarRacing-v3", continuous=True)
env = DummyVecEnv([lambda: env]) # SB3 requires a vectorized env

# Use the 'CnnPolicy' because the input is images
model = SAC(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./sac_tensorboard/",
    buffer_size=100000, # Use a larger buffer for SAC
    learning_rate=3e-4
)

print("Training SAC expert...")
# Train for a long time. 50,000 steps is just a quick test.
# A good expert will need 500k-1M+ steps.
model.learn(total_timesteps=500_000)

model.save("sac_car_racing_expert")
print("Expert model saved to sac_car_racing_expert.zip")
