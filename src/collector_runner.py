# import traci
import libsumo as traci
from pathlib import Path

from simulation_runner import SimulationRunner
from base_sumo_env import BaseSumoEnvironment
from data_collector import DataCollector
import logging

logger = logging.getLogger("v2x")

class RunnerWithCollector(SimulationRunner):
    def __init__(self, *args, collector, **kwargs):
        super().__init__(*args, **kwargs)
        self.collector = collector

    def run_until_end(self):
        self.env.step(0)
        current_time = traci.simulation.getTime()
        vehicle_count = traci.vehicle.getIDCount()
        logger.info(f"Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")
        self.collector.collect(current_time)

        try:
            while traci.simulation.getMinExpectedNumber() != 0 and vehicle_count != 0:
                self.env.step(0)
                current_time = traci.simulation.getTime()
                vehicle_count = traci.vehicle.getIDCount()
                logger.info(f"Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")
                self.collector.collect(current_time)

            logger.info("Simulation ended naturally.")

        except traci.exceptions.FatalTraCIError as e:
            logger.error(f"Fatal TraCI error occurred. Ending simulation: {e}")

    def run_with_steps(self):
        for step in range(self.simulation_steps):
            self.env.step(0)
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()
            logger.info(f"Step {step}: Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")
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
    args = SimulationRunner.parse_arguments()
    cfg = resolve_config()

    baseline_collector = DataCollector(batch_size=1000, reset_on_start=True)
    run_once(
        config_path=cfg,
        steps=args.steps,
        gui=args.gui,
        bsm=False, tls=False, priority=False, reroute=False,
        collector=baseline_collector
    )

    src_dir = Path(__file__).resolve().parent
    csv_path = src_dir / "data" / "vehicles.csv"
    baseline_path = src_dir / "data" / "vehicles_baseline.csv"
    if csv_path.exists():
        csv_path.rename(baseline_path)

    params_collector = DataCollector(batch_size=1000, reset_on_start=True)
    run_once(
        config_path=cfg,
        steps=args.steps,
        gui=args.gui,
        bsm=args.bsm, tls=args.tls, priority=args.priority, reroute=args.reroute,
        collector=params_collector
    )
if __name__ == "__main__":
    main()
