import pandas as pd
import traci
import logging
from pathlib import Path

class DataCollector:
    def __init__(self, batch_size=1000, reset_on_start=True):
        src_dir = Path(__file__).resolve().parent
        self.out_dir = src_dir / "data"
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
            is_stopped_now = speed_now < 0.1

            if is_stopped_now and not self.was_stopped[vid]:
                self.stops[vid] += 1

            self.was_stopped[vid] = is_stopped_now

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

