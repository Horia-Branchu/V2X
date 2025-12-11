import numpy as np
import gymnasium as gym
import logging
import sys
import libsumo as traci
from terminal_display import terminal_display
from base_v2x_feature import BaseV2XFeature
from gymnasium.spaces import Discrete, Box

# feature-level logger; routine per-step data goes to DEBUG, important events should use INFO
logger = logging.getLogger("v2x.features")


class DynamicTLS(BaseV2XFeature):

    def __init__(self, feature_name="DynamicTLS", enabled=True):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.observation_size = 5       # dummy observation size
        self.action_size = 4            # Extend for NS/EW, switch or maintain
        self.detection_range = 50.0     # meters
        self.extend_time = 5.0          # seconds
        self.lane_last_green = {}       # {lane_id: last_time_green}
        self.tls_last_switch = {}       # {tls_id: last_switch_timestamp}
        self.max_wait = 30.0            # maximum amount of seconds a lane can wait
        self._tls_log_events = []       # per-step event buffer for compact TTY display or verbose non-TTY logs
        self.phase_time = 0
        self._last_phase = {}

        self.min_detection_range = 20.0
        self.max_detection_range = 100.0

        self.min_extend_time = 2.0
        self.max_extend_time = 10.0

        self.w_wait = 1.0
        self.w_queue = 0.5
        self.w_switch = 2.0

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
        return gym.spaces.Dict({
        "tl_action": Discrete(4),       
        "params": Box(low=0, high=1, shape=(2,), dtype=np.float32)
    })

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
        self.lane_last_green[lane_id] = traci.simulation.getTime()

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

    # Main dynamic TLS control controller:
    # - detects vehicles approaching intersections
    # - extends green phase of the TLS if a vehicle is approaching from that side
    # - if there are no vehicles approaching the green phase of the TLS but other lanes do, switch to the next phase
    # - restores default TLS program after manual overrides
    def dynamic_tls(self, tls_id):

        current_time = traci.simulation.getTime()
        phase = traci.trafficlight.getPhase(tls_id)
        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        phases = program.phases
        
        if tls_id not in self.tls_last_switch:
            self.tls_last_switch[tls_id] = current_time

        time_in_phase = current_time - self.tls_last_switch[tls_id]

        approaching = self.get_approaching_vehicles_by_lane(tls_id)
        edge_count = {edge: len(v_list) for edge, v_list in approaching.items()}
        total_approaching = sum(edge_count.values())

        state = traci.trafficlight.getRedYellowGreenState(tls_id)
        controlled = traci.trafficlight.getControlledLinks(tls_id)

        green_lanes = []
        for i, links in enumerate(controlled):
            if i < len(state) and state[i] in ("G", "g"):
                for inc, out, via in links:
                    green_lanes.append(inc)

        cars_in_green = sum(len(approaching.get(lane, [])) for lane in green_lanes)
        cars_elsewhere = total_approaching - cars_in_green

        # 1. If lane hasn't had green for more than max_wait, force if
        if approaching:
            starved_lane = None
            max_wait_elapsed = -1.0
            for lane in approaching.keys():
                last = self.lane_last_green.get(lane, 0.0)
                wait_elapsed = current_time - last
                if wait_elapsed > max_wait_elapsed:
                    max_wait_elapsed = wait_elapsed
                    starved_lane = lane

            if starved_lane and max_wait_elapsed >= self.max_wait:
                target_phase_idx = None
                for p_idx, p in enumerate(phases):
                    p_state = p.state
                    for link_idx, links in enumerate(controlled):
                        if link_idx < len(p_state) and p_state[link_idx] in ("G", "g"):
                            if any(link[0] == starved_lane for link in links):
                                target_phase_idx = p_idx
                                break
                    if target_phase_idx is not None:
                        break

                if target_phase_idx is not None:
                    traci.trafficlight.setPhase(tls_id, int(target_phase_idx))
                    traci.trafficlight.setPhaseDuration(tls_id, self.extend_time * 2)
                    self.tls_last_switch[tls_id] = current_time
                    new_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                    new_controlled = traci.trafficlight.getControlledLinks(tls_id)
                    for i, links in enumerate(new_controlled):
                        if i < len(new_state) and new_state[i] in ("G", "g"):
                            for inc, _, _ in links:
                                self.lane_last_green[inc] = current_time
                    self.spat_message_log(
                        f"Starvation: switched phase {phase} -> {target_phase_idx} to serve {starved_lane} (wait={max_wait_elapsed:.1f}s)",
                        event_type="FORCED_GREEN",
                    )
                    return
                else:
                    tls_state = list(traci.trafficlight.getRedYellowGreenState(tls_id))
                    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
                    for i, links in enumerate(controlled_links):
                        lane_found = any(link[0] == starved_lane for link in links)
                        tls_state[i] = 'G' if lane_found else 'r'
                    traci.trafficlight.setRedYellowGreenState(tls_id, sel.extend_time)
                    self.tls_last_switch[tls_id] = current_time
                    self.lane_last_green[starved_lane] = current_time
                    self.spat_message_log(
                        f"Starvation: RG override for {starved_lane} (wait={max_wait_elapsed:.1f}s) — fallback",
                        event_type="FORCED_GREEN",
                    )
                    return


        # 2. Extend green when vehicles are approaching on it
        if cars_in_green > 0:
            traci.trafficlight.setPhaseDuration(tls_id, self.extend_time)
            self.spat_message_log(
                f"{cars_in_green} cars on green → extending",
                event_type="EXTEND_GREEN"
            )
            return

        # 3. Green has no cars but others are waiting
        if cars_in_green == 0 and cars_elsewhere > 0:
            next_phase = (phase + 1) % len(phases)
            traci.trafficlight.setPhase(tls_id, next_phase)
            traci.trafficlight.setPhaseDuration(tls_id, self.extend_time)
            self.tls_last_switch[tls_id] = current_time

            new_state = traci.trafficlight.getRedYellowGreenState(tls_id)
            new_controlled = traci.trafficlight.getControlledLinks(tls_id)
            for i, links in enumerate(new_controlled):
                if i < len(new_state) and new_state[i] in ("G", "g"):
                    for inc, _, _ in links:
                        self.lane_last_green[inc] = current_time

            self.spat_message_log(
                f"No cars on green but {cars_elsewhere} cars elsewhere → switching phase",
                event_type="SWITCH_GREEN"
            )
            return

        

            ''''''

        # 4. Force switch if green phase has been on for too long
        if time_in_phase >= self.extend_time:
            next_phase = (phase + 1) % len(phases)
            traci.trafficlight.setPhase(tls_id, next_phase)
            traci.trafficlight.setPhaseDuration(tls_id, self.extend_time)
            self.tls_last_switch[tls_id] = current_time

            new_state = traci.trafficlight.getRedYellowGreenState(tls_id)
            new_controlled = traci.trafficlight.getControlledLinks(tls_id)
            for i, links in enumerate(new_controlled):
                if i < len(new_state) and new_state[i] in ("G", "g"):
                    for inc, _, _ in links:
                        self.lane_last_green[inc] = current_time

            self.spat_message_log(
                f"Phase {phase} exceeded max green → switching to {next_phase}",
                event_type="FORCED_SWITCH"
            )
            return
    
    def _appply_rl_action(self, tls_id, tl_action):
        current_phase = traci.trafficlight.getPhase(tls_id)
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        phase_count = len(logic.phases)

        if tl_action == 0:
            if current_phase in [0, 1]:  
                traci.trafficlight.setPhaseDuration(tls_id, float(self.extend_time))

        elif tl_action == 1:
            if current_phase in [2, 3]:  
                traci.trafficlight.setPhaseDuration(tls_id, float(self.extend_time)) 

        elif tl_action == 2:
            pass  
        elif tl_action == 3:
            next_phase = (current_phase + 1) % phase_count
            traci.trafficlight.setPhase(tls_id, next_phase)

    def take_action(self, action):
        # clear per-step buffer
        self._tls_log_events.clear()

        tls_list = traci.trafficlight.getIDList()
        rl_mode = isinstance(action, dict)

        if rl_mode:
            tl_action = action["tl_action"]
            alpha, beta = action["params"]

            self.detection_range = (self.min_detection_range + float(alpha) * (self.max_detection_range - self.min_detection_range))
            self.extend_time = (self.min_extend_time + float(beta) * (self.max_extend_time - self.min_extend_time))
            
            for tls_id in tls_list:
                self._appply_rl_action(tls_id, tl_action)
        else:
            for tls_id in tls_list:
                self.dynamic_tls(tls_id)

        # Emit aggregated output
        self._log_tls_events()

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

           if len(obs) == 0:
               return np.array([0.0], dtype=np.float32 )

       return np.array(obs, dtype=np.float32)

    def calculate_reward(self):
        tls_list = traci.trafficlight.getIDList()
        total_waiting = 0.0
        total_queue = 0.0
        switches = 0
        bonus = 0.0

        current_phase = {}
        for tls_id in tls_list:
            phase = traci.trafficlight.getPhase(tls_id)
            current_phase[tls_id] = phase
            last_phase = self._last_phase.get(tls_id, phase)
            if phase != last_phase:
                switches += 1

            detection_range = getattr(self, 'detection_range', self.detection_range)
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            lane_queues = {}
            for lane in lanes:
                vehs = traci.lane.getLastStepVehicleIDs(lane)
                halting_count = traci.lane.getLastStepHaltingNumber(lane)
                total_queue += halting_count

                lane_waiting = 0.0
                for v in vehs:
                    try:
                        pos = traci.vehicle.getLanePosition(v)
                        lane_len = traci.lane.getLength(lane)
                        dist = max(0.0, lane_len - pos)
                    except Exception:
                        dist = 0.0

                    if 0 <= dist <= detection_range:
                        lane_waiting += traci.vehicle.getWaitingTime(v)

                total_waiting += lane_waiting
                lane_queues[lane] = halting_count

            if lane_queues:
                max_queue_lane = max(lane_queues, key=lane_queues.get)
                min_queue_lane = min(lane_queues, key=lane_queues.get)
                max_queue = lane_queues[max_queue_lane]
                min_queue = lane_queues[min_queue_lane]

                if self.is_lane_green(tls_id, max_queue_lane):
                    bonus += 0.5 * max_queue  

                elif self.is_lane_green(tls_id, min_queue_lane) and max_queue - min_queue > 5:
                    bonus += 0.3 * max_queue  
        
        self._last_phase = current_phase

        reward = -(self.w_wait * total_waiting + self.w_queue * total_queue + self.w_switch * switches) + bonus        
        logger.debug(f"[{self.feature_name}] reward parts: "
                     f"waiting = {total_waiting:.1f}, queue = {total_queue:.1f}, switches = {switches} "
                     f"=> reward = {reward:.3f}")
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
        self.lane_last_green.clear()
        self.tls_last_switch.clear()
        logger.debug(f"[{self.feature_name}] Reset")
