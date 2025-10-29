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

        """Resetting each time we run the program"""
        if reset_on_start and self.csv_path.exists():
            self.csv_path.unlink()

    def collect(self, t):
        """Collect data from SUMO at time t"""
        frame = []
        for vid in traci.vehicle.getIDList():
            frame.append({
                "time": t,
                "veh_id": vid,
                "edge": traci.vehicle.getRoadID(vid),
                "lane": traci.vehicle.getLaneID(vid),
                "pos": traci.vehicle.getLanePosition(vid),
                "speed": traci.vehicle.getSpeed(vid),
                "accel": traci.vehicle.getAcceleration(vid),
                "angle": traci.vehicle.getAngle(vid),
                "waiting": traci.vehicle.getWaitingTime(vid),
            })
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