#!/usr/bin/env python3

import os
import logging

from base_sumo_env import BaseSumoEnvironment

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
        if not issubclass(BaseSumoEnvironment, sumo_env):
            raise TypeError("env_class must be a subclass of BaseSumoEnvironment")

        self.env = sumo_env(config_path, **kwargs)

    def start_simulation(self):
        """Start the SUMO simulation with TraCI"""
        self.env.reset()

        if self.env.simulation_steps is not None:
            self.env.run_steps(self.env.simulation_steps)
        else:
            self.env.run_until_end()

        self.env.close()

def main():
    # import here the class that implements BaseSumoEnvironment
    from your_concrete_env import ConcreteSumoEnvironment  # Import your actual implementation

    # Parse command line arguments
    args = BaseSumoEnvironment.parse_arguments()

    # Path to SUMO configuration
    script_dir = os.path.dirname(__file__)
    sumo_config = os.path.join(script_dir, '..', 'config', 'simulation.sumocfg')

    # create simulation runner
    runner = SimulationRunner(
        sumo_env=ConcreteSumoEnvironment,
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
