import numpy as np
import gymnasium as gym
import logging
import libsumo as traci
from base_v2x_feature import BaseV2XFeature

# feature-level logger; routine per-step data goes to DEBUG, important events should use INFO
logger = logging.getLogger("v2x.features")


class DynamicTLS(BaseV2XFeature):

    def __init__(self, feature_name="DynamicTLS", enabled=True):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.observation_size = 3       # dummy observation size
        self.action_size = 2            # dummy action size
        self.detection_range = 50       # meters
        self.extend_time = 5            # seconds
        self.tls_override_times = {}    # {tls_id: timestamp}

    def get_observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(self.observation_size,))

    def get_action_space(self):
        return gym.spaces.Discrete(self.action_size)

    # Groups approaching vehicles by lane within the detection range
    def get_approaching_vehicles_by_lane(self, tls_id):
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        approaching = {}

        for lane_id in lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            lane_length = traci.lane.getLength(lane_id)
            near_tls = []

            for v_id in vehicle_ids:
                vehicle_pos = traci.vehicle.getLanePosition(v_id)
                if lane_length - vehicle_pos < self.detection_range:
                    near_tls.append(v_id)
            
            if near_tls:
                approaching[lane_id] = near_tls

        return approaching

    # Returns all lanes belonging to a strret
    def get_lanes_on_same_street(self, tls_id, lane_id):
        edge_id = traci.lane.getEdgeID(lane_id)
        all_lanes = traci.trafficlight.getControlledLanes(tls_id)
        same_street_lanes = [lane for lane in all_lanes if traci.lane.getEdgeID(lane) == edge_id]
        return same_street_lanes

    #Sets TLS light for named vehicle to green
    def set_tls_green_for_vehicle(self, tls_id, v_id):
        lane_id = traci.vehicle.getLaneID(v_id)
        street_lanes = self.get_lanes_on_same_street(tls_id,lane_id)
        tls_state = list(traci.trafficlight.getRedYellowGreenState(tls_id))
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        for i, links in enumerate(controlled_links):
            lane_found = any(link[0] in street_lanes for link in links)
            tls_state[i] = 'G' if lane_found else 'r'

        traci.trafficlight.setRedYellowGreenState(tls_id,''.join(tls_state))

    # Checks if the TLS light for the named lane is green
    def is_lane_green(self, tls_id,lane_id):
        tls_state = traci.trafficlight.getRedYellowGreenState(tls_id)
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        for i, links in enumerate(controlled_links):
            if lane_id in [link[0] for link in links]:
                if tls_state[i].lower() != 'g':
                    return False
        return True
    
    # Prints SPaT (Signal Phase and Timing) messages to console
    def spat_message_log(self,message):
        timestamp = traci.simulation.getTime()
        log_message = f"[{timestamp:.1f}s] {message}"
        logger.info(log_message)

    # Main dynamic TLS control function:
    # - detects vehicles approaching intersections
    # - extends geern lights dynamically
    # - switches light to green if only one direction has vehicles
    # - grants green light to fewer cars that are waiting for a lot of cars to pass
    # - restores default TLS program after manual overrides
    def dynamic_tls(self, tls_id):

        current_time = traci.simulation.getTime()
        tls_lanes = traci.trafficlight.getControlledLanes(tls_id)
        vehicle_list = traci.vehicle.getIDList()

        # Checks if an override is already active and reverts to normal if time expired
        if tls_id in self.tls_override_times:
            if current_time - self.tls_override_times[tls_id] >= self.extend_time:
                traci.trafficlight.setProgram(tls_id,"0")
                self.spat_message_log(f"TLS {tls_id} returning to normal program")
                del self.tls_override_times[tls_id]

        approaching = self.get_approaching_vehicles_by_lane(tls_id)
        if not approaching:
            return

        for v_id in vehicle_list:
            lane_id = traci.vehicle.getLaneID(v_id)
            if lane_id not in tls_lanes:
                continue

            distance_to_tls = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(v_id)

            if distance_to_tls < self.detection_range:

                remaining = traci.trafficlight.getNextSwitch(tls_id) - current_time
            
                # Case 1: Extend already green light
                if self.is_lane_green(tls_id, lane_id) and remaining < self.extend_time:
                    traci.trafficlight.setPhaseDuration(tls_id, self.extend_time)
                    self.spat_message_log(f"Vehicle {v_id} approaching {tls_id}, extending GREEN for {self.extend_time}s.")
                    return
                
                # Case 2: Turn green if only one lane is approaching
                if len(approaching) == 1 and lane_id in approaching:
                    if not self.is_lane_green(tls_id, lane_id):
                        self.set_tls_green_for_vehicle(tls_id, v_id)
                        traci.trafficlight.setPhaseDuration(tls_id, self.extend_time)
                        self.tls_override_times[tls_id] = current_time
                        self.spat_message_log(f"Only vehicles on lane {lane_id} near {tls_id}, switching to GREEN")
                    return
                
                # Case 3: Turn light green if there is only one vehicle waiting for a lot of vehicles to pass
                edge_counts = {edge: len(v_list) for edge, v_list in approaching.items()}

                if edge_counts:
                    max_edge = max(edge_counts, key = edge_counts.get)
                    min_edge = min(edge_counts, key = edge_counts.get)

                    max_count = edge_counts[max_edge]
                    min_count = edge_counts[min_edge]

                    if (min_count > 0 and min_count <= 3) and max_count - min_count > 5:
                        v_id = approaching[min_edge][0]
                        if not self.is_lane_green(tls_id,lane_id):
                            self.set_tls_green_for_vehicle(tls_id,v_id)
                            traci.trafficlight.setPhaseDuration(tls_id, self.extend_time)
                            self.tls_override_times[tls_id] = current_time
                            self.spat_message_log(f"Imbalance detected at {tls_id}, granting short green for lane {min_edge}")
        return
    
    def take_action(self, action):

        for tls_id in traci.trafficlight.getIDList():
            self.dynamic_tls(tls_id)

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
