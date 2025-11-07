import gymnasium as gym
import traci
import logging
import platform
import subprocess
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BaseSumoEnvironment(gym.Env):
    def __init__(self, sumo_config, simulation_steps=1000, gui=True,
                 bsm=False, tls=False, priority=False, reroute=False):
        super().__init__()

        self.sumo_config = sumo_config
        self.simulation_steps = simulation_steps
        self.current_step = 0
        self.gui = gui

        # feature management
        self.features = []
        self._setup_features(bsm, tls, priority, reroute)

        # initialize spaces
        self.observation_space = None
        self.action_space = None
        self._setup_spaces()

        self.sumo_cmd = self._build_sumo_command()

    # to be implemented in the future of RL
    def _setup_spaces(self):
        pass

    # crucial step for V2X people
    # when you implement a feature, import it here and add it to the feature
    # space ( I left an example for BSM )
    def _setup_features(self, bsm, tls, priority, reroute):
        """Initialize features based on flags"""
        self.features = []
        # if bsm:
            # from features.bsm_feature import BSMFeature
            # self.features.append(BSMFeature())
        # if tls:
        # if priority:
        # if reroute:

    def _build_sumo_command(self):
        base_cmd = ["sumo-gui" if self.gui else "sumo", "-c", self.sumo_config]
        return base_cmd

    def reset(self, seed=None, options=None):
        try:
            traci.close()
        except Exception:
            pass

        traci.start(self.sumo_cmd)
        self.current_step = 0

        for feature in self.features:
            feature.feature_reset()

        self._scenario_reset()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # distribute action to features
        self._take_action(action)

        # advance simulation
        traci.simulationStep()
        self.current_step += 1

        # linear stepping for each feature
        for feature in self.features:
            feature.feature_step()

        # collect results
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _take_action(self, action):
        """Distribute action to appropriate features"""
        if not self.features:
            return

        action_idx = 0
        for feature in self.features:
            feature_action = action[action_idx] if isinstance(action, (list, np.ndarray)) else action
            feature.take_action(feature_action)
            action_idx += 1

    def _get_observation(self):
        """Combine observations from all features"""
        if not self.features:
            return np.array([0])

        obs_parts = []
        for feature in self.features:
            feature_obs = feature.get_observation()
            if isinstance(feature_obs, (list, np.ndarray)):
                obs_parts.extend(feature_obs)
            else:
                obs_parts.append(feature_obs)

        return np.array(obs_parts)

    def _calculate_reward(self):
        """Combine rewards from all features"""
        # right now it's a dummy computation (will be implemented in the future)
        total_reward = 0
        for feature in self.features:
            total_reward += feature.calculate_reward() * feature.weight
        return total_reward

    def _scenario_reset(self):
        """Override this for specific scenario setup"""
        pass

    def _is_terminated(self):
        """Override this for scenario-specific termination"""
        return False

    def _is_truncated(self):
        return self.current_step >= self.simulation_steps

    def _get_info(self):
        info = {
            "step": self.current_step,
            "active_features": [f.get_feature_name() for f in self.features],
        }
        return info

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
