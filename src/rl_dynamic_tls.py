import numpy as np
import gymnasium as gym
import libsumo as traci 
import logging

from base_sumo_env import BaseSumoEnvironment
logger = logging.getLogger("v2x.features")

class RLDynamicTLS(BaseSumoEnvironment):
    def __init__(self, feature_name="RL_DynamicTLS", enable=True):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.observation_size = 5
        self.action_size = 4
        self.phase_time = 0
        self.extended_sex = 5

    def get_observation _space(self):
        return gym.spaces.Box(
            low=0,
            high=200,
            shape=(self.observation_size,),
            dtype=np.float32
        )

    def get_action_space(self):
        return gym.spaces.Discrete(self.action_size) 

    def get_observation(self):
        obs_list=[]

        for tls in traci.trafficlight.getIDList():
            lanes=traci.trafficlight.getControlledLanes(tls)

            try:
                qN=traci.lane.getLastStepHaltingNumber(lanes[0])
                qS=traci.lane.getLastStepHaltingNumber(lanes[1])
                qE=traci.lane.getLastStepHaltingNumber(lanes[2])
                qW=traci.lane.getLastStepHaltingNumber(lanes[3])
            except IndexError:
                qN=0
                qS=0
                qE=0
                qW=0

            obs_list.extend([qN,qS,qE,qW,float(self.phase_time)])

        obs_arry=np.array(obs_list, dtype=np.float32)
        logger.debug(f"[{self.feature_name}] Observation = {obs_arry}")
        return obs_arry

    def calculate_reward(self):
        total_waiting_time=0.0
        for vih in traci.vehicle.getIDList():
            total_waiting_time+=traci.vehicle.getWaitingTime(vih)
        
        reward=-total_waiting_time  
        
        logger.debug(f"[{self.feature_name}] Reward = {reward:.3f}")
        return reward
