import numpy as np
import gymnasium as gym
import logging
import sys
import libsumo as traci
from ui.terminal_display import terminal_display
from features.base_v2x_feature import BaseV2XFeature

logger = logging.getLogger("v2x.features")


class BSMFeature(BaseV2XFeature):

    def __init__(self, feature_name="BSMFeature", enabled=True, rl_mode=False):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.rl_mode = rl_mode      # True for RL, False for rule-based
        # Observation: [count, avg_speed, avg_wait] * 4 approaches = 12
        self.observation_size = 12
        
        # Core safety parameters
        self.detection_range = 50.0     # meters
        self.ttc_threshold = 1.5        # seconds  
        self.brake_duration = 1.0       # seconds
        
        # Parameter ranges for RL
        self.min_detection_range = 20.0
        self.max_detection_range = 100.0
        self.min_ttc_threshold = 0.5
        self.max_ttc_threshold = 3.0
        
        # Reward weights
        self.w_ttc = 2.0
        self.w_brake = 0.5
        self.w_critical = 5.0
        self.w_wait = 1.0  # New: Weight for waiting time reduction
        
        # Per-step event buffer
        self._bsm_log_events = []
        
        # Persistent state
        self._last_brake_step = {}
        self._emergency_brake_count = 0
        self._slowdown_count = 0

    def get_observation_space(self):
        # 4 approaches * 3 metrics (count, speed, wait)
        return gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.observation_size,),
            dtype=np.float32
        )

    def get_action_space(self):
        # Dummy action space: 5 discrete actions
        return gym.spaces.Discrete(5)

    def get_observation(self):
        if not self.enable:
            return np.zeros(self.observation_size, dtype=np.float32)
        
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except traci.TraCIException as e:
            logger.warning(f"[{self.feature_name}] Failed to get vehicle list: {e}")
            return np.zeros(self.observation_size, dtype=np.float32)
        
        # Approaches: N, E, S, W buckets
        # N: 315-45, E: 45-135, S: 135-225, W: 225-315
        approach_stats = np.zeros((4, 3), dtype=np.float32)
        
        for vid in vehicle_ids:
            try:
                angle = traci.vehicle.getAngle(vid)
                speed = traci.vehicle.getSpeed(vid)
                wait = traci.vehicle.getAccumulatedWaitingTime(vid)
                
                # Determine bucket (0=N, 1=E, 2=S, 3=W)
                # Sumo angle: 0 is North, 90 is East, etc.
                if angle < 45 or angle >= 315:
                    bucket = 0
                elif 45 <= angle < 135:
                    bucket = 1
                elif 135 <= angle < 225:
                    bucket = 2
                else: # 225 <= angle < 315
                    bucket = 3
                
                approach_stats[bucket][0] += 1.0
                approach_stats[bucket][1] += speed
                approach_stats[bucket][2] += wait
                
            except traci.TraCIException:
                continue

        # Average aggregation
        # [count, avg_speed, avg_wait] for each approach
        obs = []
        for i in range(4):
            count = approach_stats[i][0]
            if count > 0:
                avg_speed = approach_stats[i][1] / count
                avg_wait = approach_stats[i][2] / count
            else:
                avg_speed = 0.0
                avg_wait = 0.0
            
            obs.extend([count, avg_speed, avg_wait])
            
        return np.array(obs, dtype=np.float32)

    def calculate_reward(self):
        if not self.enable:
            return 0.0
        
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except traci.TraCIException:
            return 0.0
        
        total_waiting_time = 0.0
        total_ttc_penalty = 0.0
        critical_count = 0
        
        for vehicle_id in vehicle_ids:
            # 1. Waiting Time Reward Component
            try:
                total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            except traci.TraCIException:
                pass

            # 2. Safety Reward Component
            try:
                leader_data = traci.vehicle.getLeader(vehicle_id, dist=self.detection_range)
                if not leader_data:
                    continue
                
                leader_id, gap = leader_data
                if leader_id is None or gap < 0:
                    continue
                
                try:
                    follower_speed = traci.vehicle.getSpeed(vehicle_id)
                    leader_speed = traci.vehicle.getSpeed(leader_id)
                except traci.TraCIException:
                    continue
                
                time_to_collision = self._calculate_time_to_collision(gap, follower_speed, leader_speed)
                
                if time_to_collision < self.ttc_threshold:
                    total_ttc_penalty += (self.ttc_threshold - time_to_collision)
                    critical_count += 1
                    
            except traci.TraCIException:
                continue
        
        # Reward function
        reward = (
            - self.w_wait * total_waiting_time / 100.0  # Normalized wait penalty
            - self.w_ttc * total_ttc_penalty
            - self.w_brake * self._emergency_brake_count
            - self.w_critical * critical_count
        )
        
        return float(reward)

    def get_feature_name(self):
        return self.feature_name

    def _calculate_time_to_collision(self, gap: float, follower_speed: float, leader_speed: float) -> float:
        relative_speed_mps = max(0.0, follower_speed - leader_speed)
        
        if relative_speed_mps > 1e-3:
            return gap / relative_speed_mps
        else:
            return float('inf')

    def _calculate_time_to_collision_capped(self, gap: float, follower_speed: float, leader_speed: float) -> float:
        rel_speed = follower_speed - leader_speed
        
        if rel_speed <= 1e-3:
            return 10.0
        
        time_to_collision = gap / rel_speed
        
        if not np.isfinite(time_to_collision) or time_to_collision > 10.0:
            return 10.0
        
        return max(0.0, time_to_collision)

    def bsm_message_log(self, message, event_type: str = "GENERIC"):
        try:
            timestamp = traci.simulation.getTime()
        except traci.TraCIException:
            timestamp = 0.0
        
        verbose = f"[{self.feature_name}] {event_type}: {message} @ {timestamp:.1f}s"
        short = message.split(',')[0]
        self._bsm_log_events.append((verbose, short))

    def _log_bsm_events(self):
        if not self._bsm_log_events:
            return

        if sys.stdout.isatty():
            event_count = len(self._bsm_log_events)
            latest_short = self._bsm_log_events[-1][1]
            summary = f"[{self.feature_name}] | events={event_count} | {latest_short}"
            terminal_display.update("BSM", summary)
            terminal_display.render()
        else:
            for verbose, _ in self._bsm_log_events:
                logger.info(verbose)

    def take_action(self, action):
        if not self.enable:
            return

        self._bsm_log_events.clear()
        self._emergency_brake_count = 0
        self._slowdown_count = 0

        # Dummy action handling for RL mode
        if self.rl_mode:
            logger.debug(f"[{self.feature_name}] Dummy Action: {action}")
            # Note: We aren't changing params based on dummy action yet
        
        try:
            vehicle_list = traci.vehicle.getIDList()
        except traci.TraCIException:
            return
            
        for vehicle_id in vehicle_list:
            self.apply_bsm_safety(vehicle_id)

        self._log_bsm_events()

    def apply_bsm_safety(self, vehicle_id):
        try:
            current_time_s = traci.simulation.getTime()
            
            leader_data = traci.vehicle.getLeader(vehicle_id, dist=self.detection_range)
            if not leader_data:
                return

            leader_id, distance_to_leader_m = leader_data
            if leader_id is None or distance_to_leader_m < 0 or distance_to_leader_m > self.detection_range:
                return

            try:
                follower_speed_mps = traci.vehicle.getSpeed(vehicle_id)
                leader_speed_mps = traci.vehicle.getSpeed(leader_id)
                leader_accel_mps2 = traci.vehicle.getAcceleration(leader_id)
            except traci.TraCIException:
                return

        except traci.TraCIException:
            return

        relative_speed_mps = max(0.0, follower_speed_mps - leader_speed_mps)
        time_to_collision_s = (
            distance_to_leader_m / relative_speed_mps if relative_speed_mps > 1e-3 else float("inf")
        )

        min_gap = 10.0
        max_time_to_collision_gap_m = 80.0
        max_decel_gap_m = 60.0
        leader_decel_threshold_mps2 = -4.0
        
        gap_trigger = (distance_to_leader_m < min_gap)
        time_to_collision_trigger = (
            distance_to_leader_m <= max_time_to_collision_gap_m
            and time_to_collision_s < self.ttc_threshold
        )
        decel_trigger = (
            distance_to_leader_m <= max_decel_gap_m
            and leader_accel_mps2 <= leader_decel_threshold_mps2
        )

        should_brake = gap_trigger or time_to_collision_trigger or decel_trigger

        trigger_reasons = []
        if gap_trigger:
            trigger_reasons.append(f"GAP<{min_gap:.1f}m")
        if time_to_collision_trigger:
            trigger_reasons.append(f"TIME_TO_COLLISION<{self.ttc_threshold:.2f}s")
        if decel_trigger:
            trigger_reasons.append(f"LEADER_DECEL<={leader_decel_threshold_mps2:.1f}m/s²")

        DEFAULT_LAST_BRAKE_TIME = -10.0
        MIN_BRAKE_INTERVAL_S = 0.5
        last_brake_time = self._last_brake_step.get(vehicle_id, DEFAULT_LAST_BRAKE_TIME)
        
        if should_brake and (current_time_s - last_brake_time) >= MIN_BRAKE_INTERVAL_S:
            time_to_collision_display = (
                f"{time_to_collision_s:.2f}s"
                if np.isfinite(time_to_collision_s)
                else "distance growing"
            )

            if gap_trigger or time_to_collision_trigger:
                self._emergency_brake_count += 1
                message = f"{vehicle_id} -> leader {leader_id} (gap={distance_to_leader_m:.1f}m, time_to_collision={time_to_collision_display}, reasons=[{', '.join(trigger_reasons)}])"
                self.bsm_message_log(message, "EMERGENCY_BRAKE")
                try:
                    traci.vehicle.slowDown(vehicle_id, 0.0, self.brake_duration)
                    self._last_brake_step[vehicle_id] = current_time_s
                except traci.TraCIException:
                    pass
            else:
                self._slowdown_count += 1
                PREEMPTIVE_SLOWDOWN_FACTOR = 0.7
                target_speed_mps = max(0.0, follower_speed_mps * PREEMPTIVE_SLOWDOWN_FACTOR)
                message = f"{vehicle_id} following {leader_id} (gap={distance_to_leader_m:.1f}m, time_to_collision={time_to_collision_display}, target={target_speed_mps:.2f}m/s, reasons=[{', '.join(trigger_reasons)}])"
                self.bsm_message_log(message, "PREEMPTIVE_SLOWDOWN")
                try:
                    traci.vehicle.slowDown(vehicle_id, target_speed_mps, self.brake_duration)
                    self._last_brake_step[vehicle_id] = current_time_s
                except traci.TraCIException:
                    pass

    def feature_step(self):
        logger.debug(f"[{self.feature_name}] Step")

    def feature_reset(self):
        self._bsm_log_events.clear()
        self._last_brake_step.clear()
        self._emergency_brake_count = 0
        self._slowdown_count = 0
        logger.debug(f"[{self.feature_name}] Reset")