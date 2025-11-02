import traci
import pandas as pd
from pathlib import Path


class DataCollector:
    def __init__(self, out_dir="data", batch_size=50, reset_on_start=True):
                self.out_dir = Path(out_dir)
                self.batch_size = batch_size
                self.buffer = []
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

                    # initialize per-vehicle timers and counters
                    if vid not in self.start_time:
                        self.start_time[vid] = t
                    if vid not in self.stops:
                        self.stops[vid] = 0
                    if vid not in self.was_stopped:
                        self.was_stopped[vid] = False


                    speed_now = traci.vehicle.getSpeed(vid)
                    accel_now = traci.vehicle.getAcceleration(vid)

                    # jerk it's basically how smoothly the acceleration is
                    dt = t - self.prev_time.get(vid, t)
                    jerk_now = (accel_now - self.prev_accel.get(vid, accel_now)) / dt if dt > 0 else 0.0
                    self.prev_accel[vid] = accel_now
                    self.prev_time[vid] = t

                    # stops counter
                    stopped_now = speed_now < 0.1
                    if stopped_now and not self.was_stopped[vid]:
                        self.stops[vid] += 1
                    self.was_stopped[vid] = stopped_now

                    trip_time_now = t - self.start_time[vid]
                    distance_now = traci.vehicle.getDistance(vid)

                    # average speed per route
                    route_id = traci.vehicle.getRouteID(vid)
                    self.route_speed_sum[route_id] = self.route_speed_sum.get(route_id, 0.0) + speed_now
                    self.route_speed_count[route_id] = self.route_speed_count.get(route_id, 0) + 1
                    route_avg_speed_now = self.route_speed_sum[route_id] / self.route_speed_count[route_id]


                    self.cumulative_co2[vid] = self.cumulative_co2.get(vid, 0.0) + co2_now

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

                if frame:
                    self.buffer.extend(frame)
                    if len(self.buffer) >= self.batch_size:
                        self.flush()

    def flush(self):
                """Save buffered data"""
                if not self.buffer:
                    return
                df = pd.DataFrame(self.buffer)
                write_header = not self.csv_path.exists()
                df.to_csv(self.csv_path, mode="a", index=False, header=write_header)
                self.buffer = []