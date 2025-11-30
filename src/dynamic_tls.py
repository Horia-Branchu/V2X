import numpy as np
import gymnasium as gym
import logging
import sys
import libsumo as traci
from terminal_display import terminal_display
from base_v2x_feature import BaseV2XFeature

# feature-level logger; routine per-step data goes to DEBUG, important events should use INFO
logger = logging.getLogger("v2x.features")


class DynamicTLS(BaseV2XFeature):

    def __init__(self, feature_name="DynamicTLS", enabled=True):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.observation_size = 5       # dummy observation size
        self.action_size = 4            # dummy action size
        self.detection_range = 50       # meters
        self.extend_time = 5            # seconds
        self.tls_override_times = {}    # {tls_id: timestamp}
        self._tls_log_events = [] # per-step event buffer for compact TTY display or verbose non-TTY logs
        self.phase_time = 0
    
    def get_observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
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
    
    # Collect SPaT (Signal Phase and Timing) messages into the per-step buffer.
    # event_type: one of EXTEND_GREEN, SWITCH_GREEN, IMBALANCE, or GENERIC
    def spat_message_log(self, message, event_type: str = "GENERIC"):
        # Build structured verbose similar to BSM feature logs and a short snippet
        timestamp = traci.simulation.getTime()
        verbose = (
            f"[{self.feature_name}] {event_type}: {message} @ {timestamp:.1f}s"
        )
        # short version for TTY display: first clause or truncated form
        short = message.split(',')[0]
        self._tls_log_events.append((verbose, short))

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
                    self.spat_message_log(
                        f"Vehicle {v_id} approaching {tls_id}, extending GREEN for {self.extend_time}s.",
                        event_type="EXTEND_GREEN",
                    )
                    return
                
                # Case 2: Turn green if only one lane is approaching
                if len(approaching) == 1 and lane_id in approaching:
                    if not self.is_lane_green(tls_id, lane_id):
                        self.set_tls_green_for_vehicle(tls_id, v_id)
                        traci.trafficlight.setPhaseDuration(tls_id, self.extend_time)
                        self.tls_override_times[tls_id] = current_time
                        self.spat_message_log(
                            f"Only vehicles on lane {lane_id} near {tls_id}, switching to GREEN",
                            event_type="SWITCH_GREEN",
                        )
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
                            self.spat_message_log(
                                f"Imbalance detected at {tls_id}, granting short green for lane {min_edge}",
                                event_type="IMBALANCE",
                            )
        return
    
    def take_action(self, action):
        #tls_list = traci.trafficlight.getIDList()
#
        #if isinstance(action, (int, float)):
        #    action = [int(action)]
        #elif isinstance(action, (list, np.ndarray)):
        #    action = [int(a) for a in action]
        #else:
        #    raise ValueError(f"Unexpected action type: {type(action)}")
        #
        #if len(action) != len(tls_list):
        #    action = [action[0]] * len(tls_list)
#
        #for idx,tls in enumerate(tls_list):
        #    act = action[idx]
        #    phase = traci.trafficlight.getPhase(tls)
        #    phase_state = traci.trafficlight.getRedYellowGreenState(tls)
        #    total_phases = len(phase_state)  
        #    phase_duration = traci.trafficlight.getPhaseDuration(tls)
#
        #    if act ==0:
        #        pass
        #    elif act ==1:
        #        traci.trafficlight.setPhaseDuration(tls, phase_duration + self.extend_time)
        #        self.spat_message_log(f"Extending GREEN for TLS {tls} by {self.extend_time}s per RL action")
#
        #    elif act ==2:
        #        next_phae = (phase +1) % total_phases
        #        traci.trafficlight.setPhase(tls,next_phae)
        #        self.spat_message_log(f"Switching TLS {tls} to next phase {next_phae} per RL action")
#
        #    elif act ==3:
        #        new_state= 'GGrr'
        #        traci.trafficlight.setRedYellowGreenState(tls,new_state)
        #        self.spat_message_log(f"Setting TLS {tls} to GREEN state per RL action")
#
        #    elif act ==4:
        #        new_state= 'rrGG'
        #        traci.trafficlight.setRedYellowGreenState(tls,new_state)
        #        self.spat_message_log(f"Setting TLS {tls} to GREEN state per RL action")

        if isinstance(action, np.ndarray):
            if action.shape == ():
                action = int(action)
            else:
                action = int(action[0])
        elif isinstance(action, list):
            action = int(action[0])
        else:
            action = int(action)
            
        for tls_id in traci.trafficlight.getIDList(): 
            phase = traci.trafficlight.getPhase(tls_id) 
            phase_duration = traci.trafficlight.getPhaseDuration(tls_id) 
            phase_state = traci.trafficlight.getRedYellowGreenState(tls_id)
            total_phases = len(phase_state)
            
            if action == 0: 
                if phase %2 == 0: 
                    traci.trafficlight.setPhaseDuration(tls_id, phase_duration + self.extend_time)
                    self.spat_message_log(f"Extending NS green at {tls_id} by {self.extend_time}s") 

            elif action == 1: 
                if phase %2 == 1: 
                    traci.trafficlight.setPhaseDuration(tls_id, phase_duration + self.extend_time) 
                    self.spat_message_log(f"Extending EW green at {tls_id} by {self.extend_time}s") 
                    
            elif action == 2: 
                traci.trafficlight.setPhase(tls_id, (phase+1) % total_phases) 
                self.spat_message_log(f"Switching {tls_id} to next phase") 

            elif action == 3: 
                pass    

        self.phase_time += 1 
        logger.debug(f"[{self.feature_name}] Action taken: {action} on TLS {tls_id}")
                

    def get_observation(self):
        obs = []

        for tls_id in traci.trafficlight.getIDList():
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            direction_groups = {}
            for lane in lanes:
                edge = lane.split('_')[0]
                if edge not in direction_groups:
                    direction_groups[edge] = []
                direction_groups[edge].append(lane)

            direction_queues = []

            for edge, grouped_lanes in direction_groups.items():
                total_q = 0
                for lane in grouped_lanes:
                    total_q += traci.lane.getLastStepVehicleNumber(lane)

                total_q = min(total_q, 20)
                direction_queues.append(total_q)

            while len(direction_queues) < 4:
                direction_queues.append(0)
            direction_queues = direction_queues[:4]

            elapsed = (traci.trafficlight.getPhaseDuration(tls_id) - traci.trafficlight.getNextSwitch(tls_id)) 
            elapsed = max(0, min(elapsed, 60))

            state_value = (
                direction_queues[0]*1 +
                direction_queues[1]*100 +
                direction_queues[2]*10000 +
                direction_queues[3]*1000000 +
                int(elapsed)*100000000
            )

            obs.append(state_value/ 1e10)
            return np.array(obs, dtype=np.float32)

    def calculate_reward(self):
        total_waiting = 0.0

        for v_id in traci.vehicle.getIDList():
            total_waiting += traci.vehicle.getWaitingTime(v_id)

        logger.debug(f"[{self.feature_name}] Reward: {total_waiting}")
        return -total_waiting

    def get_feature_name(self):
        return self.feature_name

    def _log_tls_events(self):

        if not self._tls_log_events:
            return

        if sys.stdout.isatty():
            tls_count = len(self._tls_log_events)
            latest_short = self._tls_log_events[-1][1]
            summary = f"[{self.feature_name}] | tls_events={tls_count} | {latest_short}"
            terminal_display.update("TLS", summary)
            terminal_display.render()
        else:
            for verbose, _ in self._tls_log_events:
                logger.info(verbose)

    def feature_step(self):
        # default behavior: don't spam the console for rule-based runs
        logger.debug(f"[{self.feature_name}] Step completed")

    def feature_reset(self):
        logger.debug(f"[{self.feature_name}] Reset")
