import libsumo as traci
import logging
import argparse
import sys

from pathlib import Path
from stable_baselines3 import PPO
from simulation_runner import SimulationRunner
from environment.base_sumo_env import BaseSumoEnvironment
from datacollector.data_collector import DataCollector, baseline_filename, data_dir_name, rl_filename
from analysis import correlation_map, geo_emissions_plot, geo_plots, plots
from collector_runner import RunnerWithCollector

logger = logging.getLogger("v2x")

class RLRunnerWithCollector(SimulationRunner):
    def __init__(self, *args, collector, model, **kwargs):
        super().__init__(*args, **kwargs)
        self.collector = collector
        self.model = model

    def run_until_end(self):
        """Run the simulation until it ends and collect data at each step"""
        obs, _ = self.env.reset()
        try:
            while traci.simulation.getMinExpectedNumber() != 0:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)

                current_time = traci.simulation.getTime()
                self.collector.collect(current_time)
                if terminated or truncated:
                    break

        except traci.exceptions.FatalTraCIError as e:
            logger.error(f"Fatal TraCI error occurred: {e}")

        self.collector.flush()

    def run_with_steps(self):
        """Run the simulation for a given number of steps while collecting data."""
        obs, _ = self.env.reset()
        try:
            for _ in range(self.simulation_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                current_time = traci.simulation.getTime()
                self.collector.collect(current_time)
                if terminated or truncated:
                    break

        except traci.exceptions.FatalTraCIError as e:
            logger.error(f"Fatal TraCI error occurred: {e}")

        self.collector.flush()

def resolve_config():
    """Using the normal config if it exists at the normal path"""
    here = Path(__file__).resolve().parent
    cfg = here.parent.parent / "config" / "simulation.sumocfg"
    if not cfg.exists():
        raise FileNotFoundError("simulation.sumocfg not found under ../config/")
    return str(cfg)


def build_env(config_path, *, gui, rl, tls, bsm, priority, reroute):
    return BaseSumoEnvironment(
        config_path,
        gui=gui,
        tls=tls,
        bsm=bsm,
        priority=priority,
        reroute=reroute,
        rl=rl,
    )


def main():
    #Argument parsing to include max points
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--max-points", type=int, default=200000)
    local_args, remaining = temp_parser.parse_known_args()
    max_points = local_args.max_points
    sys.argv = [sys.argv[0]] + remaining

    #RL arguments
    parser = argparse.ArgumentParser(description="Baseline and RL data collector")
    parser.add_argument("--tls", action="store_true", help="Run TLS RL")
    parser.add_argument("--bsm", action="store_true", help="Run BSM RL")
    parser.add_argument("--priority", action="store_true", help="Run Priority RL")
    parser.add_argument("--reroute", action="store_true", help="Run Reroute RL")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    #Enforce that at least one V2X feature is selected for RL
    enabled_features = []
    if args.tls:
        enabled_features.append("tls")
    if args.bsm:
        enabled_features.append("bsm")
    if args.priority:
        enabled_features.append("priority")
    if args.reroute:
        enabled_features.append("reroute")

    if len(enabled_features) == 0:
        raise ValueError(f"No arguments parsed. Please parse exactly one argument")
    if len(enabled_features) > 1:
        raise ValueError(f"Please specify exactly one feature for RL collector!\nPlease choose exactly one of the following --tls --bsm --priority --reroute --gui")

    feature_name = enabled_features[0]

    #Paths and output files
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / data_dir_name
    data_dir.mkdir(exist_ok=True)

    baseline_path = data_dir / baseline_filename
    rl_path = data_dir / rl_filename

    if baseline_path.exists():
        raise ValueError(f"Baseline file exists. Delete {baseline_path} first.")
    if rl_path.exists():
        raise ValueError(f"RL file exists. Delete {rl_path} first.")

    cfg = resolve_config()

    logger.info("Running BASELINE simulation")

    #Baseline run with no V2X or RL features enabled
    baseline_env = build_env(cfg, gui=args.gui, rl=False, tls=False, bsm=False, priority=False, reroute=False)
    baseline_collector = DataCollector(
        output_filename=baseline_filename,
        reset_on_start=True,
    )

    baseline_runner = RunnerWithCollector(
        cfg,
        sumo_env=baseline_env,
        steps=args.steps,
        collector=baseline_collector,
    )
    baseline_runner.start_simulation()

    #RL run with exactly one feature
    logger.info(f"Running RL simulation for feature: {feature_name}")
    model_path = project_root / f"{feature_name}_feature_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"RL model not found: {model_path}")

    #load model
    model = PPO.load(model_path)

    rl_env = build_env(
        cfg,
        gui=args.gui,
        rl=True,
        tls=args.tls,
        bsm=args.bsm,
        priority=args.priority,
        reroute=args.reroute,
    )

    rl_collector = DataCollector(
        output_filename=rl_path,
        reset_on_start=True,
    )

    rl_runner = RLRunnerWithCollector(
        cfg,
        sumo_env=rl_env,
        steps=args.steps,
        collector=rl_collector,
        model=model,
    )
    rl_runner.start_simulation()


    print("\nGenerating plots...")
    plots.main(max_points=max_points)
    correlation_map.main()
    geo_plots.main()
    geo_emissions_plot.main()
    print("All plots generated successfully")


if __name__ == "__main__":
    main()
