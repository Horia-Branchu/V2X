import gymnasium as gym
import traci
from abc import ABC, abstractmethod
import logging
import platform
import subprocess
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BaseSumoEnvironment(gym.Env, ABC):
    def __init__(self, sumo_config, simulation_steps=1000, gui=True,
                 bsm=False, tls=False, priority=False, reroute=False):
        super().__init__()

        self.sumo_config = sumo_config
        self.simulation_steps = simulation_steps
        self.current_step = 0
        self.gui = gui

        # modes
        self.bsm = bsm
        self.tls = tls
        self.priority = priority
        self.reroute = reroute

        self.sumo_cmd = self._build_sumo_command()

    def _build_sumo_command(self):
        base_cmd = ["sumo-gui" if self.gui else "sumo", "-c", self.sumo_config]
        return base_cmd

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

    # the none used arguments should be placed for the RL manifold
    def reset(self, seed=None, options=None):
        try:
            traci.close()
        except Exception as e:
            logging.error(f"Traci could not be closed: {e}")
            pass

        traci.start(self.sumo_cmd)
        self.current_step = 0

        self._check_unimplemented_features()

        self._scenario_reset()

        # initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def stepon(self, action):

        self._take_action(action)

        # advance simulation
        traci.simulationStep()
        self.current_step += 1

        # collect the results
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        try:
            traci.close()
        except Exception as e:
            logging.error(f"Traci could not be closed: {e}")
            pass

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

    @staticmethod
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

    # here are the abstract methods that will be implemented for the
    # specific scenarios we will be working in
    @abstractmethod
    def _take_action(self, action):
        raise NotImplementedError("execute the given action in the simulation")

    @abstractmethod
    def _get_observation(self):
        raise NotImplementedError("get the current observation from the simulation")

    @abstractmethod
    def _calculate_reward(self):
        raise NotImplementedError("compute the reward based on current state")

    # optional hooks with default implementations
    def _scenario_reset(self):
        """Scenario-specific reset logic"""
        raise NotImplementedError("scenario specific (maybe?) reset logic")

    def _is_terminated(self):
        raise NotImplementedError("check if the run should terminate (collision?)")

    def _is_truncated(self):
        """Check if episode is truncated by step limit"""
        return self.current_step >= self.simulation_steps

    def _get_info(self):
        raise NotImplementedError("get additional info for debugging")
