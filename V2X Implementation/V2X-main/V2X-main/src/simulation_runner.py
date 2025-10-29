#!/usr/bin/env python3

import traci
import os
import argparse
import logging
import subprocess
import platform
from DataCollector import DataCollector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SimulationRunner:
    def __init__(self, config_path, gui=False, bsm=False, tls=False, priority=False, reroute=False):
        self.config_path = config_path
        self.gui = gui
        self.bsm = bsm
        self.tls = tls
        self.priority = priority
        self.reroute = reroute
        self.DataCollector = DataCollector(out_dir="data")

    def start_simulation(self):
        """Start the SUMO simulation with TraCI"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        traci.start([sumo_binary, "-c", self.config_path])

    def run_until_end(self):
        """Run simulation until it naturally ends"""
        # Check for unimplemented features
        self._check_unimplemented_features()

        traci.simulationStep()
        vehicle_count = traci.vehicle.getIDCount()
        try:
            while (traci.simulation.getMinExpectedNumber() != 0 and vehicle_count != 0):
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                vehicle_count = traci.vehicle.getIDCount()
                self.DataCollector.collect(current_time)
                print(f"Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")
            print("Simulation ended naturally.")
                
        except traci.exceptions.FatalTraCIError:
            logging.error("Fatal TraCI error occurred. Ending simulation.")
        

    def run_steps(self, num_steps):
        """Run simulation for a specified number of steps"""
        # Check for unimplemented features
        self._check_unimplemented_features()

        for step in range(num_steps):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()
            print(f"Step {step}: Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")

    def _check_unimplemented_features(self):
        """Log warnings for features that are enabled but not yet implemented"""
        if self.bsm:
            logging.warning("BSM (Basic Safety Message) feature is not yet implemented")
        if self.tls:
            logging.warning("TLS (Traffic Light System) control feature is not yet implemented")
        if self.priority:
            logging.warning("Priority vehicle handling feature is not yet implemented")
        if self.reroute:
            logging.warning("Dynamic rerouting feature is not yet implemented")

    def close_simulation(self):
        """Close the TraCI connection and SUMO"""
        traci.close()
        if self.gui:
            # Force close SUMO GUI
            try:
                if platform.system() == "Windows": # they're special like that
                    subprocess.run(["taskkill", "/F", "/IM", "sumo-gui.exe"], check=False, capture_output=True)
                else:  # Linux and others
                    subprocess.run(["pkill", "-f", "sumo-gui"], check=False, capture_output=True)
                print("SUMO GUI closed.")
            except Exception as e:
                print(f"Warning: Could not close SUMO GUI: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run SUMO simulation with TraCI")
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of steps to run the simulation (default: run until simulation ends)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SUMO in GUI mode instead of CLI mode (default: CLI)"
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
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Path to SUMO configuration
    script_dir = os.path.dirname(__file__)
    sumo_config = os.path.join(script_dir, '..', 'config', 'simulation.sumocfg')

    # Create simulation runner
    runner = SimulationRunner(
        sumo_config,
        gui=args.gui,
        bsm=args.bsm,
        tls=args.tls,
        priority=args.priority,
        reroute=args.reroute
    )

    # Start simulation
    runner.start_simulation()

    # Run simulation based on arguments
    if args.steps is not None:
        runner.run_steps(args.steps)
    else:
        runner.run_until_end()

    # Close simulation
    runner.close_simulation()

if __name__ == "__main__":
    main()
