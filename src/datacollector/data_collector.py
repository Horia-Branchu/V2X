import pandas as pd
import libsumo as traci
from pathlib import Path

# global definition
baseline_filename = "vehicle_state_baseline.parquet"
v2x_filename = "vehicle_state_v2x.parquet"
rule_based_filename = "vehicle_state_rule_based.parquet"
rl_filename = "vehicle_state_rl.parquet"
data_dir_name = "data"
DEFAULT_MAX_POINTS = None

class DataCollector:
    def __init__(self,output_filename,reset_on_start=True):
        project_root = Path(__file__).resolve().parents[2]
        self.out_dir = project_root / data_dir_name
        self.buffer = []
        # making the directory in a file named data
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.output_filename = output_filename
        self.parquet_path = self.out_dir / self.output_filename
        # dict for feature engineering
        self.prev_accel = {}
        self.prev_time = {}
        self.was_stopped = {}
        self.stops = {}
        self.queue_time = {}

        """Reset file each time the program starts
           CLOSE THE parquet FILE IF IT'S OPENED BEFORE RERUNNING"""
        if reset_on_start and self.parquet_path.exists():
            self.parquet_path.unlink()

    # typevar is important to be specified in this method
    def collect(self, time: float):
        """Collect data from SUMO at time t"""
        frame = []
        for vid in traci.vehicle.getIDList():
            co2_now = traci.vehicle.getCO2Emission(vid)
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

            is_stopped_now = speed_now < 0.1
            #count stop event only when transitioning from moving
            if is_stopped_now and not self.was_stopped[vid]:
                self.stops[vid] += 1
            self.was_stopped[vid] = is_stopped_now

            if vid not in self.queue_time:
                self.queue_time[vid] = 0.0

            if is_stopped_now:
                self.queue_time[vid] += time_diff

            frame.append({
                "time": time,
                "veh_id": vid,
                "edge": traci.vehicle.getRoadID(vid),
                "speed": speed_now,
                "accel": accel_now,
                "co2": co2_now,
                "jerk": jerk_now,
                "stops": self.stops[vid],
                "time_loss": traci.vehicle.getTimeLoss(vid),
                "queue_time": self.queue_time[vid],
            })

        if len(frame) > 0:
            self.buffer.extend(frame)

    def flush(self):
        """Save buffered data"""
        if len(self.buffer) == 0:
            return None

        df = pd.DataFrame(self.buffer)
        df.to_parquet(self.parquet_path,index=False)
        self.buffer = []
