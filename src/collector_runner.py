import libsumo as traci
import logging
import argparse
import sys

from pathlib import Path
from simulation_runner import SimulationRunner
from base_sumo_env import BaseSumoEnvironment
from data_collector import DataCollector, vehicle_filename
from analysis import correlation_map, geo_plots, plots

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
    parent_path = here.parent / "config" / "simulation.sumocfg"
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
    #Extend argument parsing to include max points
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Maximum number of sampled points used in plotting"
    )
    local_args, remained_argv = temp_parser.parse_known_args()
    max_points = local_args.max_points
    sys.argv = [sys.argv[0]] + remained_argv
    args = SimulationRunner.parse_arguments()
    cfg = resolve_config()

    if not(args.bsm or args.tls or args.priority or args.reroute):
         raise ValueError(f"\nNo v2x features enabled\n"
         "At least one of --bsm --tls --priority --reroute must be true")

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / vehicle_filename
    baseline_path = project_root / "data" / f"{Path(vehicle_filename).stem}_baseline.csv"
    # Ensuring baseline and v2x files do not exist
    if baseline_path.exists():
        raise ValueError(f"\nBaseline file exists.Delete {baseline_path} before rerunning")
    elif csv_path.exists():
        raise ValueError(f"\nV2X file exists.Delete {csv_path} before rerunning")

    baseline_collector = DataCollector(batch_size=10000, reset_on_start=True)
    run_once(
        config_path=cfg,
        steps=args.steps,
        gui=args.gui,
        bsm=False, tls=False, priority=False, reroute=False,
        collector=baseline_collector
    )

    if csv_path.exists():
        csv_path.rename(baseline_path)

    params_collector = DataCollector(batch_size=10000, reset_on_start=True)
    run_once(
        config_path=cfg,
        steps=args.steps,
        gui=args.gui,
        bsm=args.bsm, tls=args.tls, priority=args.priority, reroute=args.reroute,
        collector=params_collector
    )

    print(f"\n\n\nGenerating Plots")
    plots.main(max_points=max_points)
    correlation_map.main()
    geo_plots.main()
    print(f"All plots generated successfully")

if __name__ == "__main__":
    main()