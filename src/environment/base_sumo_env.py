import gymnasium as gym
import libsumo as traci
import logging
import sys
from ui.terminal_display import terminal_display
import platform
import subprocess
import numpy as np
import threading
import itertools
import sys
import time
from features.dummy_feature import DummyFeature
from features.dynamic_tls import DynamicTLS
from features.bsm_feature import BSMFeature
from features.priority_corridor import PriorityCorridorFeature

# use a named logger for the project; features can log at DEBUG for RL and INFO for rule-based
logger = logging.getLogger("v2x")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)

class BaseSumoEnvironment(gym.Env):
    def __init__(self, sumo_config, gui=True,
                 bsm=False, tls=False, priority=False, reroute=False, rl=False):
        super().__init__()

        self.sumo_config = sumo_config
        self.current_step = 0
        self.gui = gui
        # whether this environment is used for RL (verbose per-step logs) or rule-based
        self.rl = rl

        # configure logger level for RL vs rule-based runs
        logger.setLevel(logging.DEBUG if self.rl else logging.INFO)

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
        if self.rl and self.features:
            if len(self.features) == 1:
                self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
                # Box action space for RL: [tl_action, alpha, beta]
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
            else:
                self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(1)
        return

    # crucial step for V2X people
    # when you implement a feature, import it here and add it to the feature
    # space ( I left dummy examples )
    def _setup_features(self, bsm, tls, priority, reroute):
        self.features = []

        if bsm:
            self.features.append(BSMFeature("BSMFeature"))

        if tls:
            self.features.append(DynamicTLS("DynamicTLS", rl_mode=self.rl))

        if priority:
            self.features.append(PriorityCorridorFeature("PriorityCorridorFeature", rl_mode=self.rl))

        if reroute:
            self.features.append(DummyFeature("RerouteFeature"))

    def _build_sumo_command(self):
        base_cmd = ["sumo-gui" if self.gui else "sumo", "-c", self.sumo_config]
        return base_cmd

    def reset(self, seed=None, options=None):
        try:
            traci.close()
        except Exception:
            pass

        stop_event = threading.Event()
        try:
            traci.start(self.sumo_cmd)
        finally:
            stop_event.set()

        self.current_step = 0

        if self.rl and len(self.features) == 1:
            feature = self.features[0]
            self.observation_space = feature.get_observation_space()
            self.action_space = feature.get_action_space()

        for feature in self.features:
            feature.feature_reset()

        self._scenario_reset()

        observation = self._get_observation()
        info = self._get_info()

        return (observation, info)

    def step(self, action):
        # distribute action to features
        self._take_action(action)

        # advance simulation
        traci.simulationStep()
        self.current_step += 1

        # log concise vehicle count per step so any caller of env.step() sees it
        try:
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()

            msg = f"Time {current_time:.1f}s: Vehicles in simulation: {vehicle_count}"

            # update the shared terminal display (TTY) or emit INFO (non-TTY)
            terminal_display.update("ENV", msg)
            terminal_display.render()

            # If simulation has ended (no expected vehicles or no vehicles present),
            # finalize the display and emit a final INFO.
            if traci.simulation.getMinExpectedNumber() == 0 or vehicle_count == 0:
                terminal_display.finish()
                logger.info("Simulation ended naturally.")
        except Exception:
            # if traci not available or hasn't started yet, skip logging
            pass

        # linear stepping for each feature
        for feature in self.features:
            feature.feature_step()

        # collect results
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _take_action(self, action):
        """Distribute action to appropriate features"""
        if not self.features:
            return

        if self.rl and len(self.features) == 1:
            self.features[0].take_action(action)
            return

        action_idx = 0
        for feature in self.features:
            if isinstance(action, (list, np.ndarray, tuple)):
                if isinstance(action, np.ndarray) and action.ndim == 0:
                    feature_action = action.item()
                else:
                    if action_idx < len(action):
                        feature_action = action[action_idx]
                    else:
                        feature_action = action[-1]
            else:
                feature_action = action

            feature.take_action(feature_action)
            action_idx += 1

    def _get_observation(self):
        """Combine observations from all features"""
        if not self.features:
            return np.array([0], dtype=np.float32)  # Specify dtype

        obs_parts = []
        for feature in self.features:
            feature_obs = feature.get_observation()
            if isinstance(feature_obs, (list, np.ndarray)):
                # Convert to float32 and flatten if needed
                feature_obs = np.array(feature_obs, dtype=np.float32).flatten()
                obs_parts.extend(feature_obs)
            else:
                # Convert scalar to float32
                obs_parts.append(np.float32(feature_obs))

        observation = np.array(obs_parts, dtype=np.float32)

        # Ensure the shape matches your observation space
        # If your observation space is (1,), make sure it has exactly 1 element
        if observation.shape != (1,):
            # Either reshape or handle accordingly
            # If you need exactly 1 element, take the first one or aggregate
            observation = np.array([observation[0]], dtype=np.float32)

        return observation

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
        try:
            vehicle_count = traci.vehicle.getIDCount()
            if vehicle_count == 0:
                logger.info("Termination: No vehicles left in simulation")
                return True
        except Exception as my_ex:
            logger.error(f"Exception caught at calling _is_terminated: {my_ex}")
            pass

        return False

    # def _is_truncated(self):
    #     return self.current_step >= self.simulation_steps

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
            logger.error(f"Traci could not be closed: {e}")
            pass

        if self.gui:
            # Force close SUMO GUI
            try:
                if platform.system() == "Windows": # they're special like that
                    subprocess.run(["taskkill", "/F", "/IM", "sumo-gui.exe"], check=False, capture_output=True)
                else:  # Linux and others
                    subprocess.run(["pkill", "-f", "sumo-gui"], check=False, capture_output=True)
                logger.info("SUMO GUI closed.")
            except Exception as e:
                logger.warning(f"Could not close SUMO GUI: {e}")