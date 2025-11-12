#!/usr/bin/env python3

import os
import argparse
import logging
import traci

from base_sumo_env import BaseSumoEnvironment

# use the project logger
logger = logging.getLogger("v2x")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)

class SimulationRunner:
    def __init__(self, config_path, sumo_env, steps, **kwargs):
        """
        Arguments:
            sumo_env: A concrete implementation of BaseSumoEnvironment
            config_path: Path to SUMO config file
            **kwargs: Additional arguments for the environment we work in
        """
        self.simulation_steps = steps
        if isinstance(sumo_env, BaseSumoEnvironment):
            # we were given an environment instance
            self.env = sumo_env
        else:
            # create environment from config
            env_class = sumo_env if sumo_env else BaseSumoEnvironment
            self.env = env_class(config_path, **kwargs)

    def run_manual_feature_test(self):
        logger.info("------ Feature testing mode ------")
        logger.info(f"active features: {[f.get_feature_name() for f in self.env.features]}")

        obs, _ = self.env.reset()

        def simulation_logic(current_step):
            action = self.env.action_space.sample()

            # step the environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                obs, _ = self.env.reset()
                logger.info(f"------ simulation reseted at step {current_step} ------")

        if self.simulation_steps is not None:
            for current_step in range(self.simulation_steps):
                simulation_logic(current_step)
        else:
            current_step = 0
            while (traci.simulation.getMinExpectedNumber() != 0):
                simulation_logic(current_step)
                current_step += 1

        self.env.close()

    def test_specific_feature(self, feature_name):
        """Test a specific feature in isolation"""
        logger.info(f"------ Isolated feature testing: {feature_name} ------")
        obs, _ = self.env.reset()

        def simulation_logic(current_step):
            # custom action logic for specific feature testing
            action = self._get_feature_specific_action(feature_name, current_step)

            obs, reward, terminated, truncated, info = self.env.step(action)

            # per-step feature test message — verbose (use DEBUG so rule-based runs don't get spammed)
            logger.debug(f"Step {current_step}: {feature_name}")

            if terminated or truncated:
                return -1

        if self.simulation_steps is not None:
            for current_step in range(self.simulation_steps):
                if simulation_logic(current_step) == -1:
                    break
        else:
            current_step = 0
            while (traci.simulation.getMinExpectedNumber() != 0):
                if simulation_logic(current_step) == -1:
                    break
                current_step += 1


        self.env.close()

    def _get_feature_specific_action(self, feature_name, step):
        """Generate specific actions for feature testing"""
        return self.env.action_space.sample()

    def run_until_end(self):
        """Run simulation until it naturally ends"""

        traci.simulationStep()
        vehicle_count = traci.vehicle.getIDCount()
        try:
            while (traci.simulation.getMinExpectedNumber() != 0 and vehicle_count != 0):
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                vehicle_count = traci.vehicle.getIDCount()
                # keep the original concise rule-based log line
                logger.info(f"Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")

            logger.info("Simulation ended naturally.")

        except traci.exceptions.FatalTraCIError as e:
            logger.error(f"Fatal TraCI error occurred. Ending simulation: {e}")

    def run_with_steps(self):
        """Run simulation for a specified number of steps"""

        for step in range(self.simulation_steps):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()
            logger.info(f"Step {step}: Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}")

    def start_simulation(self):
        """Start the SUMO simulation with TraCI"""
        self.env.reset()

        if self.simulation_steps is not None:
            self.run_with_steps()
        else:
            self.run_until_end()

        self.env.close()

    @staticmethod
    def parse_arguments():
        """Parse command line arguments for feature testing"""
        parser = argparse.ArgumentParser(description="Run SUMO simulation with feature testing")
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
        return parser.parse_args()

def main():
    args = SimulationRunner.parse_arguments()

    script_dir = os.path.dirname(__file__)
    sumo_config = os.path.join(script_dir, '..', 'config', 'simulation.sumocfg')

    env = BaseSumoEnvironment(
        sumo_config,
        gui=args.gui,
        bsm=args.bsm,
        tls=args.tls,
        priority=args.priority,
        reroute=args.reroute
    )

    from priority_corridor import PriorityCorridorRunner
    runner = PriorityCorridorRunner(sumo_config, steps=args.steps, sumo_env=env)

    enabled_features = []
    if args.bsm: enabled_features.append("bsm")
    if args.tls: enabled_features.append("tls")
    if args.priority: enabled_features.append("priority")
    if args.reroute: enabled_features.append("reroute")

    # run appropriate testing based on arguments
    if args.test_all:
        runner.run_manual_feature_test()
    elif len(enabled_features) == 1:
        runner.test_specific_feature(enabled_features[0])
    elif len(enabled_features) > 1:
        logger.info(f"Multiple features enabled: {enabled_features}, using manual test mode")
        runner.run_manual_feature_test()
    else:
        runner.start_simulation()

if __name__ == "__main__":
    main()