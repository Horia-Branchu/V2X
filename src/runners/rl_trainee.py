import os
import argparse
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from environment.base_sumo_env import BaseSumoEnvironment

class StopAtTimeCallback(BaseCallback):
    def __init__(self, stop_time, verbose=1):
        super().__init__(verbose)
        self.stop_time = stop_time

    def _on_step(self) -> bool:
        if datetime.datetime.now() >= self.stop_time:
            print("\n=== STOP TIME REACHED. ENDING TRAINING ===")
            return False  
        return True
    
def main(args, sumo_config="config/simulation.sumocfg"):
    enabled_features = []
    if args.bsm: enabled_features.append("bsm")
    if args.tls: enabled_features.append("tls")
    if args.priority: enabled_features.append("priority")

    if len(enabled_features) == 0:
        print("ERROR: Please specify one feature for RL!")
        exit(68)

    if len(enabled_features) > 1:
        print("ERROR: Please specify only one feature at a time!")
        exit(67)

    feature_name = enabled_features[0]

    print(f"=== {feature_name.upper()} RL TEST STARTED ===\n")

    # create environment with ONLY feature selected by dev
    env = BaseSumoEnvironment(
        sumo_config,
        gui=args.gui,
        tls=args.tls,
        bsm=args.bsm,
        priority=args.priority,
        reroute=False,
        rl=True
    )

    start_time = datetime.datetime.now()
    # Set duration to 10 hours
    stop_time = start_time + datetime.timedelta(hours=10)

    callback = StopAtTimeCallback(stop_time)

    # train RL agent for this feature
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        # Sumo and traci runs only on CPU so it stands to reason that we should not use GPU
        device='cpu'
    )

    print(f"Training {feature_name} feature...")

    # Save a checkpoint every 100,000 steps (~1 hour at 28 fps)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./logs/",
        name_prefix=f"{feature_name}_model"
    )

    # Combine callbacks
    callback_list = CallbackList([callback, checkpoint_callback])

    model.learn(total_timesteps=10_000_000_000, callback=callback_list)
    model.save(f"{feature_name}_feature_model")

    # test the trained feature
    print(f"Testing {feature_name} feature...")
    obs, _ = env.reset()
    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    env.close()
    print(f"=== {feature_name.upper()} RL TEST COMPLETE ===\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rl modules")
    parser.add_argument("--tls",  action="store_true", help="Run simulation_runner", default=False)
    parser.add_argument("--bsm", action="store_true", help="Run data_collector", default=False)
    parser.add_argument("--priority", action="store_true", help="Run priority RL", default=False)
    parser.add_argument("--gui", action="store_true", help="Run with GUI", default=False)

    main(args=parser.parse_args())
