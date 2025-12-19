import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from stable_baselines3 import PPO
from base_sumo_env import BaseSumoEnvironment

def main():
    parser = argparse.ArgumentParser(description="Run RL features")
    parser.add_argument("--tls", action="store_true", help="Run TLS feature")
    parser.add_argument("--bsm", action="store_true", help="Run BSM feature")
    parser.add_argument("--gui", action="store_true", help="Start simulation in GUI")
    args = parser.parse_args()

    enabled_features = []
    if args.bsm: enabled_features.append("bsm")
    if args.tls: enabled_features.append("tls")

    if len(enabled_features) == 0:
        print(f"ERROR: Please specify one feature for RL testing!\nPlease choose one of the following --tls --bsm --gui")
        exit(68)

    if len(enabled_features) > 1:
        print("ERROR: Please specify only one feature at a time testing!")
        exit(67)


    # Create environment
    env = BaseSumoEnvironment(
        "config/simulation.sumocfg",
        gui=args.gui,
        tls=args.tls,
        bsm=args.bsm,
        priority=False,
        reroute=False,
        rl=True
    )

    feature_name = enabled_features[0]

    # load model
    model = None
    try:
        model = PPO.load(f"{feature_name}_feature_model")
        print(f"Loaded {feature_name} model")
    except:
        print(f"No {feature_name} model, make sure you have the model, exiting!")
        exit(56)

    # run simulation until it ends
    obs, _ = env.reset()
    total_reward = 0
    step = 0

    print("Starting simulation...")

    while True:
        action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if terminated or truncated:
            print(f"Simulation ended at step {step}")
            break

    env.close()
    print(f"final reward: {total_reward:.3f}")
    print(f"total steps: {step}")

if __name__ == "__main__":
    main()
