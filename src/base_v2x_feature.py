from abc import ABC, abstractmethod
import gymnasium as gym # used in the future

class BaseV2XFeature(ABC):
    def __init__(self, enabled=True):
        self.enable = enabled
        self.weight = 1.0

    @abstractmethod
    def get_observation_space(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def take_action(self, action):
        pass

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def calculate_reward(self):
        pass

    def feature_step(self):
        pass

    def feature_reset(self):
        pass
