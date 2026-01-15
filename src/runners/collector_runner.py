import libsumo as traci
import logging
import argparse
import sys

from pathlib import Path
from runners.simulation_runner import SimulationRunner
from environment.base_sumo_env import BaseSumoEnvironment
from datacollector.data_collector import DataCollector, baseline_filename, v2x_filename, rule_based_filename, data_dir_name, DEFAULT_MAX_POINTS
from analysis import correlation_map, geo_emissions_plot, geo_plots, plots

logger = logging.getLogger("v2x")

class RunnerWithCollector(SimulationRunner):
    def __init__(self, *args, collector, **kwargs):
        super().__init__(*args, **kwargs)
        self.collector = collector

    def run_until_end(self):
        self.env.step(0)
        current_time = traci.simulation.getTime()
        vehicle_count = traci.vehicle.getIDCount()
        self.collector.collect(current_time)

        try:
            while traci.simulation.getMinExpectedNumber() != 0 and vehicle_count != 0:
                self.env.step(0)
                current_time = traci.simulation.getTime()
                vehicle_count = traci.vehicle.getIDCount()
                self.collector.collect(current_time)

            # collector flush/closing handled by caller or run_with_steps

        except traci.exceptions.FatalTraCIError as e:
            logger.error(f"Fatal TraCI error occurred. Ending simulation: {e}")

        self.collector.flush()

    def run_with_steps(self):
        for step in range(self.simulation_steps):
            self.env.step(0)
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()
            self.collector.collect(current_time)
        self.collector.flush()

def resolve_config():
    """Using the normal config if it exists at the normal path"""
    here = Path(__file__).resolve().parent
    parent_path = here.parent.parent / "config" / "simulation.sumocfg"
    if parent_path.exists():
        return str(parent_path)

    raise FileNotFoundError("No .sumocfg found. Pass --config or place simulation.sumocfg under ..\\config\\")

def build_env(config_path, *, gui, bsm, tls, priority, reroute):
    """Create the same BaseSumoEnvironment"""
    return BaseSumoEnvironment(
        config_path,
        gui=gui,
        bsm=bsm,
        tls=tls,
        priority=priority,
        reroute=reroute
    )

# typevar is important to be specified in this method
def run_once(config_path: str, steps: int | None, gui: bool,
             bsm: bool, tls: bool, priority: bool, reroute: bool,
             collector: DataCollector):
    env = build_env(config_path, gui=gui, bsm=bsm, tls=tls, priority=priority, reroute=reroute)
    runner = RunnerWithCollector(config_path, sumo_env=env, steps=steps, collector=collector)
    runner.start_simulation()

def main():
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS ,
        help="Maximum number of sampled points used in plotting",
    )
    temp_parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="Regenerate baseline parquet even if it already exists.",
    )
    temp_parser.add_argument(
        "--force-rule-based",
        action="store_true",
        help="Regenerate rule-based parquet even if it already exists.",
    )

    local_args, remained_argv = temp_parser.parse_known_args()
    max_points = local_args.max_points
    force_baseline = local_args.force_baseline
    force_rule = local_args.force_rule_based

    sys.argv = [sys.argv[0]] + remained_argv
    args = SimulationRunner.parse_arguments()
    cfg = resolve_config()

    if not (args.bsm or args.tls or args.priority or args.reroute):
        raise ValueError(
            "\nNo v2x features enabled\n"
            "At least one of --bsm --tls --priority must be true"
        )

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / data_dir_name
    baseline_path = data_dir / baseline_filename
    rule_path = data_dir / rule_based_filename

    if baseline_path.exists() and force_baseline:
        baseline_path.unlink()

    if not baseline_path.exists():
        baseline_collector = DataCollector(
            output_filename=baseline_filename,
            reset_on_start=True,
        )
        run_once(
            config_path=cfg,
            steps=args.steps,
            gui=args.gui,
            bsm=False,
            tls=False,
            priority=False,
            reroute=False,
            collector=baseline_collector,
        )
    else:
        print(f"Baseline exists, reusing: {baseline_path}")

    if rule_path.exists() and force_rule:
        rule_path.unlink()

    if not rule_path.exists():
        rule_collector = DataCollector(
            output_filename=rule_based_filename,
            reset_on_start=True,
        )
        run_once(
            config_path=cfg,
            steps=args.steps,
            gui=args.gui,
            bsm=args.bsm,
            tls=args.tls,
            priority=args.priority,
            reroute=args.reroute,
            collector=rule_collector,
        )
    else:
        print(f"Rule-based exists, reusing: {rule_path}")
if __name__ == "__main__":
    main()