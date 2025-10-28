import gymnasium as gym
import traci
from abc import ABC, abstractmethod

class BaseSumoEnvironment(gym.Env, ABC):
    def __init__(self, sumo_config, simulation_steps=1000, gui=True):
        super().__init__()

        # configuration merging with Mihai?
        self.sumo_config = sumo_config
        self.simulation_steps = simulation_steps
        self.current_step = 0
        self.gui = gui

        self.sumo_cmd = self._build_sumo_command()

    def _build_sumo_command(self):
        base_cmd = ["sumo-gui" if self.gui else "sumo", "-c", self.sumo_config]
        return base_cmd

    def reset(self, seed=None, options=None):
        try:
            traci.close()
        except:
            pass

        traci.start(self.sumo_cmd)
        self.current_step = 0

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
        except:
            pass

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
