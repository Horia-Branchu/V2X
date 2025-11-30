import numpy as np
import gymnasium as gym
import libsumo as traci 
import logging


logger = logging.getLogger("v2x.features")

class RLDynamicTLS():
    def __init__(self, feature_name="RL_DynamicTLS", enable=True):
        super().__init__(enable)
        self.feature_name = feature_name
        self.observation_size = 5
        self.action_size = 4
        self.phase_time = 0
        self.extended_sec = 5

    def get_observation_space(self):
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

    def take_action(self, action):
        tls_list=traci.trafficlight.getIDList()

        if not isinstance(action, (list, tuple, np.ndarray)):
            action = [action]
        if len(action) != len(tls_list):
            logger.warning(
                f"[{self.feature_name}] Action size {len(action)} does not match "
                f"TLS count {len(tls_list)}. Broadcasting first action.")
            action = [action[0]] * len(tls_list)

        for i,tls in enumerate(tls_list):
            action = action[i]
            current_pahase=traci.trafficlight.getPhase(tls) 

            if action == 0:
                pass

            elif action == 1:
                state = traci.trafficlight.getRedYellowGreenState(tls)
                if "G" in state:
                    remaining=traci.trafficlight.getPhaseDuration(tls)
                    traci.trafficlight.setPhaseDuration(tls, remaining + self.extended_sec)

            elif action == 2:   
                traci.trafficlight.setPhaseDuration(tls, 0)
                self.phase_time[tls]=0

            elif action == 3:
                next_phase = (current_pahase+1) %traci.trafficlight.getPhaseNumber(tls)
                traci.trafficlight.setPhase(tls, next_pase)
                self.phase_time[tls]=0

            else:
                logger.warning(f"[{self.feature_name}] Invalid action: {action}")
            
            self.phase_time[tls] += 1
            next_phase=traci.trafficlight.getPhase(tls)
            logger.debug(f"[{self.feature_name}] Action taken: {action} on TLS: {tls},"
                         f" from phase {current_pahase} to {next_phase}, time: {self.phase_time[tls]}")

    def get_feature_name(self):
        return self.feature_name

    def feature_step(self):
        pass

    def feature_reset(self):
        self.phase_time = 0
        logger.debug(f"[{self.feature_name}] Reset")
