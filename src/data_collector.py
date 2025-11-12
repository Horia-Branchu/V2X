
import pandas as pd
import os
import traci
from pathlib import Path
import argparse

from simulation_runner import SimulationRunner
from base_sumo_env import BaseSumoEnvironment
import logging
logger = logging.getLogger("v2x")

class DataCollector:
    def __init__(self, out_dir="data", batch_size=1000, reset_on_start=True):
                self.out_dir = Path(out_dir)
                self.batch_size = batch_size
                self.buffer = []
                # making the directory in a file named Data
                self.out_dir.mkdir(parents=True, exist_ok=True)
                self.csv_path = self.out_dir / "vehicles.csv"
                # dict for feature engineering
                self.cumulative_co2 = {}
                self.prev_accel = {}
                self.prev_time = {}
                self.start_time = {}
                self.was_stopped = {}
                self.stops = {}
                self.route_speed_sum = {}
                self.route_speed_count = {}

                """Reset file each time the program starts
                   CLOSE THE CSV FILE IF IT'S OPENED BEFORE RERUNNING"""
                if reset_on_start and self.csv_path.exists():
                    self.csv_path.unlink()

    # typevar is important to be specified in this method
    def collect(self, time: float):
                """Collect data from SUMO at time t"""
                frame = []
                for vid in traci.vehicle.getIDList():
                    co2_now = traci.vehicle.getCO2Emission(vid)
                    #first time seeing this vehicle and we store its start time
                    if vid not in self.start_time:
                        self.start_time[vid] = time
                    # initialize stop counter for this vehicle
                    if vid not in self.stops:
                        self.stops[vid] = 0.0
                    #track if the vehicle was stopped in the previous step
                    if vid not in self.was_stopped:
                        self.was_stopped[vid] = False

                    speed_now = traci.vehicle.getSpeed(vid)
                    accel_now = traci.vehicle.getAcceleration(vid)

                    # jerk it's basically how smoothly the acceleration/brake is
                    time_diff  = time - self.prev_time.get(vid, time)
                    if time_diff  > 0.0:
                         jerk_now = (accel_now - self.prev_accel.get(vid, accel_now)) / time_diff
                    else:
                         jerk_now = 0.0

                    self.prev_accel[vid] = accel_now
                    self.prev_time[vid] = time

                    # stops counter
                    if speed_now < 0.1:
                        stopped_now = True
                    else:
                        stopped_now = False
                    if stopped_now and not self.was_stopped[vid]:
                        self.stops[vid] += 1

                    self.was_stopped[vid] = stopped_now

                    trip_time_now = time - self.start_time[vid]
                    distance_now = traci.vehicle.getDistance(vid)

                    # average speed per route_id
                    route_id = traci.vehicle.getRouteID(vid)
                    if route_id in self.route_speed_sum:
                        self.route_speed_sum[route_id] += speed_now
                    else:
                        self.route_speed_sum[route_id] = speed_now
                    #no of route id
                    if route_id in self.route_speed_count:
                        self.route_speed_count[route_id] += 1
                    else:
                        self.route_speed_count[route_id] = 1
                    #Average speed per route
                    route_avg_speed_now = self.route_speed_sum[route_id] / self.route_speed_count[route_id]

                    #calculating co2 per vehicle
                    if vid in self.cumulative_co2:
                        self.cumulative_co2[vid] = self.cumulative_co2[vid] + co2_now
                    else:
                        self.cumulative_co2[vid] = co2_now

                    frame.append({
                        "time": time,
                        "veh_id": vid,
                        "edge": traci.vehicle.getRoadID(vid),
                        "lane": traci.vehicle.getLaneID(vid),
                        "pos": traci.vehicle.getLanePosition(vid),
                        "speed": speed_now,
                        "accel": accel_now,
                        "angle": traci.vehicle.getAngle(vid),
                        "waiting": traci.vehicle.getWaitingTime(vid),
                        "co2": co2_now,
                        "co2_cumulative": self.cumulative_co2[vid],
                        "route_avg_speed": route_avg_speed_now,
                        "jerk": jerk_now,
                        "stops": self.stops[vid],
                        "trip_time": trip_time_now,
                        "distance": distance_now,
                    })

                if len(frame) > 0:
                    self.buffer.extend(frame)
                    if len(self.buffer) >= self.batch_size:
                        self.flush()

    def flush(self):
                """Save buffered data"""
                if len(self.buffer) == 0:
                    return None

                df = pd.DataFrame(self.buffer)
                write_header = not self.csv_path.exists()

                df.to_csv(self.csv_path, mode="a", index=False, header=write_header)
                self.buffer = []

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
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()
            logger.info(f"Step {step}: Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")
            self.collector.collect(current_time)
        self.collector.flush()

def parse_arguments():
    args = SimulationRunner.parse_arguments()

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--out-dir", type=str, default="data", help="Output directory for CSVs")
    extras, _ = extra.parse_known_args()

    args.out_dir = extras.out_dir
    return args

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
    args = parse_arguments()
    cfg = resolve_config()
    out_dir = args.out_dir

    baseline_collector = DataCollector(out_dir=out_dir, batch_size=1000, reset_on_start=True)
    run_once(
        config_path=cfg,
        steps=args.steps,
        gui=args.gui,
        bsm=False, tls=False, priority=False, reroute=False,
        collector=baseline_collector
    )

    csv_path = Path(out_dir) / "vehicles.csv"
    baseline_path = Path(out_dir) / "vehicles_baseline.csv"
    if csv_path.exists():
        csv_path.rename(baseline_path)

    params_collector = DataCollector(out_dir=out_dir, batch_size=1000, reset_on_start=True)
    run_once(
        config_path=cfg,
        steps=args.steps,
        gui=args.gui,
        bsm=args.bsm, tls=args.tls, priority=args.priority, reroute=args.reroute,
        collector=params_collector
    )
if __name__ == "__main__":
    main()