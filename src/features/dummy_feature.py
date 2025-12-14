import numpy as np
import gymnasium as gym
import logging
from features.base_v2x_feature import BaseV2XFeature

# feature-level logger; routine per-step data goes to DEBUG, important events should use INFO
logger = logging.getLogger("v2x.features")


class DummyFeature(BaseV2XFeature):
    def __init__(self, feature_name="DummyFeature", enabled=True):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.observation_size = 3  # dummy observation size
        self.action_size = 2       # dummy action size

    def get_observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(self.observation_size,))

    def get_action_space(self):
        return gym.spaces.Discrete(self.action_size)

    def take_action(self, action):
        # verbose per-step logs at DEBUG level (useful for RL)
        logger.info(f"[{self.feature_name}] Action taken: {action}")
        # no actual implementation

    def get_observation(self):
        dummy_obs = [0.1, 0.2, 0.3]  # dummy observation data
        logger.debug(f"[{self.feature_name}] Observation: {dummy_obs}")
        return np.array(dummy_obs)

    def calculate_reward(self):
        dummy_reward = 0.5  # dummy reward
        logger.debug(f"[{self.feature_name}] Reward: {dummy_reward}")
        return dummy_reward

    def get_feature_name(self):
        return self.feature_name

    def feature_step(self):
        # default behavior: don't spam the console for rule-based runs
        logger.debug(f"[{self.feature_name}] Step completed")

    def feature_reset(self):
        logger.debug(f"[{self.feature_name}] Reset")
