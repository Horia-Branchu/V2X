import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from base_sumo_env import BaseSumoEnvironment

# create the test enviroment ( will be merged with mihai )
env = BaseSumoEnvironment("config/simulation.sumocfg", gui=False) # this will error out
check_env(env)  # Validate your environment

# init the model
model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            # Sumo and traci runs only on CPU so it stands to reason that we should not use GPU
            device='cpu')

# train
model.learn(total_timesteps=100000)

# save the model
model.save("sumo_base_ppo")

# test the trained model
obs, _ = env.reset()
for _ in range(2):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
