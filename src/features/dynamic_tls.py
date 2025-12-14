import numpy as np
import gymnasium as gym
import logging
import sys
import libsumo as traci
from terminal_display import terminal_display
from features.base_v2x_feature import BaseV2XFeature

# feature-level logger; routine per-step data goes to DEBUG, important events should use INFO
logger = logging.getLogger("v2x.features")


class DynamicTLS(BaseV2XFeature):

    def __init__(self, feature_name="DynamicTLS", enabled=True, rl_mode=False):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.rl_mode = rl_mode  # True for RL, False for rule-based
        self.observation_size = 5  # Observation elements per traffic light
        self.detection_range = 50.0  # meters
        self.extend_time = 5.0  # seconds
        self.tls_override_times = {}    # {tls_id: timestamp}
        self._tls_log_events = [] # per-step event buffer for compact TTY display or verbose non-TTY logs
        self.phase_time = 0
        self._last_phase = {}

        self.min_detection_range = 20.0
        self.max_detection_range = 100.0

        self.min_extend_time = 2.0
        self.max_extend_time = 10.0

        self.w_wait = 2.0
        self.w_queue = 1.0
        self.w_switch = 5.0

        self._queue_clamp = 20
    
    def get_observation_space(self):
        tls_list = traci.trafficlight.getIDList()
        tls_count = len(tls_list)
        low = np.array([0.0]*(tls_count*self.observation_size), dtype=np.float32)
        high = []
        for _ in range(tls_count):
            high.extend([self._queue_clamp,self._queue_clamp,self._queue_clamp,self._queue_clamp,60.0])
        high = np.array(high, dtype=np.float32) if high else np.array([0.0], dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, shape=(tls_count*self.observation_size,), dtype=np.float32)
    
    def get_action_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

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
        # clear per-step buffer
        self._tls_log_events.clear()
        tls_list = traci.trafficlight.getIDList()

        if self.rl_mode:
            alpha, beta = self._parse_rl_action(action)
            
            self.detection_range = self.min_detection_range + alpha * (self.max_detection_range - self.min_detection_range)
            self.extend_time = self.min_extend_time + beta * (self.max_extend_time - self.min_extend_time)
            
            logger.debug(f"[{self.feature_name}] RL params - detection_range={self.detection_range:.1f}m, extend_time={self.extend_time:.2f}s")
        
        for tls_id in tls_list:
            self.dynamic_tls(tls_id)

        # Emit aggregated output
        self._log_tls_events()
    
    def _parse_rl_action(self, action):
        if isinstance(action, np.ndarray):
            action_flat = action.flatten()
            if len(action_flat) >= 2:
                return float(action_flat[1]), float(action_flat[2])
            elif len(action_flat) == 1:
                return float(action_flat[0]), 0.5
        elif isinstance(action, (list, tuple)) and len(action) >= 2:
            return float(action[1]), float(action[2])
        
        return 0.5, 0.5

    def get_observation(self):
        obs = []
        directions = ["N", "S", "E", "W"]
        tls_list = traci.trafficlight.getIDList()

        for tls_id in tls_list:
            lanes = traci.trafficlight.getControlledLanes(tls_id)

            groups = {d: [] for d in directions}
            for lane in lanes:
                edge = lane.split("_")[0]
                for d in directions:
                    if edge.startswith(d):
                        groups[d].append(lane)

            for d in directions:
                q = sum(traci.lane.getLastStepVehicleNumber(l) for l in groups[d])
                obs.append(min(q, self._queue_clamp))

            elapsed = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
            elapsed = max(0.0, min(float(elapsed), 60.0))
            obs.append(elapsed)

        return np.array(obs, dtype=np.float32) if obs else np.array([0.0], dtype=np.float32)

    def calculate_reward(self):
        tls_list = traci.trafficlight.getIDList()
        total_waiting = 0.0
        total_queue = 0.0
        switches = 0
        bonus = 0.0
        throughput = 0.0
        efficiency_bonus = 0.0

        current_phase = {}
        for tls_id in tls_list:
            phase = traci.trafficlight.getPhase(tls_id)
            current_phase[tls_id] = phase
            last_phase = self._last_phase.get(tls_id, phase)
            if phase != last_phase:
                switches += 1

            detection_range = getattr(self, 'detection_range', self.detection_range)
            extend_time = getattr(self, 'extend_time', self.extend_time)
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            lane_queues = {}
            green_lanes_served = 0
            
            for lane in lanes:
                vehs = traci.lane.getLastStepVehicleIDs(lane)
                halting_count = traci.lane.getLastStepHaltingNumber(lane)
                total_queue += halting_count

                lane_waiting = 0.0
                detected_in_range = 0
                
                for v in vehs:
                    try:
                        pos = traci.vehicle.getLanePosition(v)
                        lane_len = traci.lane.getLength(lane)
                        dist = max(0.0, lane_len - pos)
                        speed = traci.vehicle.getSpeed(v)
                    except Exception:
                        dist = 0.0
                        speed = 0.0

                    if 0 <= dist <= detection_range:
                        detected_in_range += 1
                        lane_waiting += traci.vehicle.getWaitingTime(v)
                        if speed > 1.0:
                            throughput += speed * 0.2

                total_waiting += lane_waiting
                lane_queues[lane] = halting_count
                
                if self.is_lane_green(tls_id, lane) and detected_in_range > 0:
                    green_lanes_served += detected_in_range

            if lane_queues:
                max_queue_lane = max(lane_queues, key=lane_queues.get)
                min_queue_lane = min(lane_queues, key=lane_queues.get)
                max_queue = lane_queues[max_queue_lane]
                min_queue = lane_queues[min_queue_lane]

                if self.is_lane_green(tls_id, max_queue_lane):
                    bonus += 1.0 * max_queue  

                elif self.is_lane_green(tls_id, min_queue_lane) and max_queue - min_queue > 5:
                    bonus += 0.5 * max_queue
                    
            efficiency_bonus += green_lanes_served * 0.3
        
        self._last_phase = current_phase

        num_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane) 
                          for tls_id in tls_list 
                          for lane in traci.trafficlight.getControlledLanes(tls_id))
        
        if num_vehicles > 0:
            avg_waiting = total_waiting / num_vehicles
        else:
            avg_waiting = 0.0

        param_efficiency = 0.0
        if 40 <= detection_range <= 80:
            param_efficiency += 1.0
        else:
            param_efficiency -= abs(detection_range - 60) * 0.1  
            
        if 4 <= extend_time <= 8:
            param_efficiency += 1.0
        else:
            param_efficiency -= abs(extend_time - 6) * 0.2  

        time_penalty = traci.simulation.getTime() * 0.001

        reward = -(self.w_wait * avg_waiting + self.w_queue * total_queue + self.w_switch * switches + time_penalty) + bonus + throughput + efficiency_bonus + param_efficiency
        
        logger.debug(f"[{self.feature_name}] reward parts: "
                     f"avg_waiting={avg_waiting:.2f}, queue={total_queue:.1f}, switches={switches}, "
                     f"bonus={bonus:.1f}, throughput={throughput:.1f}, efficiency={efficiency_bonus:.1f}, "
                     f"param_eff={param_efficiency:.2f}, time_penalty={time_penalty:.2f} => reward={reward:.3f}")
        return float(reward)

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
        det = getattr(self, "detection_range", self.detection_range)
        ext = getattr(self, "extend_time", self.extend_time)
        logger.debug(f"[{self.feature_name}] Step | detection_range={det:.1f}m extend_time={ext:.2f}s")

    def feature_reset(self):
        self._tls_log_events.clear()
        self._last_phases = {tls_id: traci.trafficlight.getPhase(tls_id) for tls_id in traci.trafficlight.getIDList()}
        self.phase_time = 0
        logger.debug(f"[{self.feature_name}] Reset")