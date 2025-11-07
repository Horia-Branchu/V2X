#!/usr/bin/env python3

import os
import logging
import traci
import pandas as pd

from base_sumo_env import BaseSumoEnvironment
from default_sumo_env import DefaultSumoEnviroment

from data_collector import DataCollector
from analysis import plots
from pathlib import Path
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SimulationRunner:
    def __init__(self, config_path, sumo_env, **kwargs):
        """
        Arguments:
            sumo_env: A concrete implementation of BaseSumoEnvironment
            config_path: Path to SUMO config file
            **kwargs: Additional arguments for the environment we work in
        """
        # fallback to default env
        if sumo_env is None:
            sumo_env = DefaultSumoEnviroment

        self.env = sumo_env(config_path, **kwargs)
        self.data_collector = DataCollector(out_dir="data")

    def run_until_end(self):
        """Run simulation until it naturally ends"""

        self.env._check_unimplemented_features()
        traci.simulationStep()
        vehicle_count = traci.vehicle.getIDCount()
        try:
            while (traci.simulation.getMinExpectedNumber() != 0 and vehicle_count != 0):
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                vehicle_count = traci.vehicle.getIDCount()
                self.data_collector.collect(current_time)
                print(f"Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")

            print("Simulation ended naturally.")

        except traci.exceptions.FatalTraCIError as e:
            logging.error(f"Fatal TraCI error occurred. Ending simulation: {e}")

    def run_steps(self, num_steps):
        """Run simulation for a specified number of steps"""

        self.env._check_unimplemented_features()

        for step in range(num_steps):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()
            self.data_collector.collect(current_time)
            print(f"Step {step}: Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")

    def start_simulation(self):
        """Start the SUMO simulation with TraCI"""
        self.env.reset()

        if self.env.simulation_steps is not None:
            self.run_steps(self.env.simulation_steps)
        else:
            self.run_until_end()
        try:
            self.data_collector.flush()
        except Exception as e:
            logging.warning(f"Final flush failed: {e}")

        #Plotting section
        try:
            #Extracting file location
            project_root = Path(__file__).resolve().parents[1]
            csv_path = plots.find_latest_csv(project_root, "vehicles.csv")
            print(f"Csv path {csv_path}\n"
                  f"Plotting data")
            #Reading the csv file
            df = pd.read_csv(csv_path, low_memory=False)
            out_dir = csv_path.parent
            plots.plot_co2_over_time(df, out_dir)
            plots.plot_accel_vs_co2(df, out_dir)
            plots.plot_speed_vs_co2(df, out_dir)
            plots.plot_min_speed_per_edge(df, out_dir)
            plots.plot_speed_over_time(df, out_dir)
            plots.plot_co2_vs_jerk(df, out_dir)
            print(f"The plots are done")
        except Exception as e:
            print(f"The occurred problem: {e}")

        self.env.close()

def main():
    # Parse command line arguments
    args = BaseSumoEnvironment.parse_arguments()

    # Path to SUMO configuration
    script_dir = os.path.dirname(__file__)
    sumo_config = os.path.join(script_dir, '..', 'config', 'simulation.sumocfg')

    # create simulation runner
    runner = SimulationRunner(
        sumo_env=None,
        config_path=sumo_config,
        simulation_steps=args.steps,
        gui=args.gui,
        bsm=args.bsm,
        tls=args.tls,
        priority=args.priority,
        reroute=args.reroute
    )

    runner.start_simulation()

if __name__ == "__main__":
    main()