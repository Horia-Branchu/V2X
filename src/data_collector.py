import traci
import pandas as pd
from pathlib import Path
import os
import traci
from base_sumo_env import BaseSumoEnvironment
import argparse

import logging
from pathlib import Path
from analysis import plots,geo_plots

class DataCollector:
    def __init__(self, out_dir="data", batch_size=50, reset_on_start=True):
                self.out_dir = Path(out_dir)
                self.batch_size = batch_size
                self.buffer = []
                # making the directory in a file named Data
                self.out_dir.mkdir(parents=True, exist_ok=True)
                self.csv_path = self.out_dir / "vehicles.csv"

                #Dict for feature engineering
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

    def collect(self, t: float):
                """Collect data from SUMO at time t"""
                frame = []
                for vid in traci.vehicle.getIDList():
                    co2_now = traci.vehicle.getCO2Emission(vid)
                    #first time seeing this vehicle and we store its start time
                    if vid not in self.start_time:
                        self.start_time[vid] = t
                    # initialize stop counter for this vehicle
                    if vid not in self.stops:
                        self.stops[vid] = 0.0
                    #track if the vehicle was stopped in the previous step
                    if vid not in self.was_stopped:
                        self.was_stopped[vid] = False

                    speed_now = traci.vehicle.getSpeed(vid)
                    accel_now = traci.vehicle.getAcceleration(vid)

                    # jerk it's basically how smoothly the acceleration/brake is
                    dt = t - self.prev_time.get(vid, t)
                    if dt > 0:
                         jerk_now = (accel_now - self.prev_accel.get(vid, accel_now)) / dt
                    else:
                         jerk_now = 0.0

                    self.prev_accel[vid] = accel_now
                    self.prev_time[vid] = t

                    # stops counter
                    if speed_now < 0.1:
                        stopped_now = True
                    else:
                        stopped_now = False
                    if stopped_now and not self.was_stopped[vid]:
                        self.stops[vid] += 1

                    self.was_stopped[vid] = stopped_now

                    trip_time_now = t - self.start_time[vid]
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
                        "time": t,
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
                if self.csv_path.exists():
                    write_header = False
                else:
                    write_header = True

                df.to_csv(self.csv_path, mode="a", index=False, header=write_header)
                self.buffer = []

    def run_loop(self, steps=None):
        traci.simulationStep()
        #I took it from the simulation_runner
        if steps is None:
          try:

            vehicle_count = traci.vehicle.getIDCount()
            while traci.simulation.getMinExpectedNumber() != 0 and vehicle_count != 0:
                current_time = traci.simulation.getTime()
                self.collect(current_time)
                print(f"Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")
                traci.simulationStep()
                vehicle_count = traci.vehicle.getIDCount()
            print(f"Simulation ended naturally.")
          except traci.exceptions.FatalTraCIError as e:
            logging.error(f"Fatal TraCI error occurred. Ending simulation: {e}")
        else:
          try:
            for step in range(steps):
                current_time = traci.simulation.getTime()
                self.collect(current_time)
                print(f"Step {step}: Time {current_time:.1f}s: Vehicles in simulation: {traci.vehicle.getIDCount()}")
                traci.simulationStep()
          except Exception as e:
              print(f"Another exception occurred {e}")

        #try except for flushing
        try:
           self.flush()
        except Exception as e:
           print(f"Flush failed {e}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SUMO DataCollector standalone")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps to run the simulation"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Run SUMO in GUI mode"
    )
    parser.add_argument(
        "--bsm",
        action="store_true",
        help="Enable Basic Safety Message (BSM) generation during the simulation"
    )
    parser.add_argument(
        "--tls",
        action="store_true",
        help="Enable Traffic Light System (TLS) control during the simulation"
    )
    parser.add_argument(
        "--priority",
        action="store_true",
        help="Enable priority vehicle handling during the simulation"
    )
    parser.add_argument(
        "--reroute",
        action="store_true",
        help="Enable dynamic rerouting of vehicles during the simulation"
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        default=False,
        help="Test all features with manual control"
    )
    args = parser.parse_args()

    # Path to SUMO configuration
    script_dir = os.path.abspath(os.path.dirname(__file__))
    sumo_config = os.path.abspath(os.path.join(script_dir, "..", "config", "simulation.sumocfg"))

    # create simulation runner
    env = BaseSumoEnvironment(
        sumo_config,
        gui=args.gui,
        bsm=args.bsm,
        tls=args.tls,
        priority=args.priority,
        reroute=args.reroute
    )
    #Initialize DataCollector
    collector = DataCollector(out_dir="data")

    #Start simulation
    env.reset()        # starts SUMO and connects TraCI
    collector.run_loop(steps=args.steps)
    try:
        collector.flush()
    except Exception as e:
        print(f"Flush failed: {e}")

    env.close()  # closes SUMO properly
    # Plotting section
    try:
        # Extracting file location
        project_root = Path(__file__).resolve().parents[1]
        csv_path = plots.find_latest_csv(project_root, "vehicles.csv")
        print(f"Csv path {csv_path}\n"
              f"Plotting data")
        # Reading the csv file
        df = pd.read_csv(csv_path, low_memory=False)
        out_dir = csv_path.parent

        #Printing the plots
        plots.plot_speed_vs_route_avg_speed(df, out_dir)
        plots.plot_accel_vs_co2(df, out_dir)
        plots.plot_speed_vs_co2(df, out_dir)
        plots.plot_co2_vs_jerk(df, out_dir)
        plots.plot_stop_duration_vs_speed(df, out_dir)
        print(f"All plots are done")

        # Geographic plot
        geo_plots.plot_min_speed_map(
            df=df,
            sumo_config=Path(env.sumo_config),
            out_path=out_dir / "min_speed_map.png",
            background_path=None,
            top_n_labels=0
        )
        print(f"The plots are done")
    except Exception as e:
        print(f"The occurred problem: {e}")