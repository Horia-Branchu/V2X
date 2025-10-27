#!/usr/bin/env python3

import traci
import os
import argparse

class SimulationRunner:
    def __init__(self, config_path, gui=False):
        self.config_path = config_path
        self.gui = gui

    def start_simulation(self):
        """Start the SUMO simulation with TraCI"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        traci.start([sumo_binary, "-c", self.config_path])

    def run_until_end(self):
        """Run simulation until it naturally ends"""
        try:
            while True:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                vehicle_count = traci.vehicle.getIDCount()
                print(f"Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")
        except traci.exceptions.FatalTraCIError:
            print("Simulation ended naturally.")

    def run_steps(self, num_steps):
        """Run simulation for a specified number of steps"""
        for step in range(num_steps):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()
            print(f"Step {step}: Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")

    def close_simulation(self):
        """Close the TraCI connection"""
        traci.close()

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
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Path to SUMO configuration
    script_dir = os.path.dirname(__file__)
    sumo_config = os.path.join(script_dir, '..', 'config', 'simulation.sumocfg')

    # Create simulation runner
    runner = SimulationRunner(sumo_config, gui=args.gui)

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
