import numpy as np
import gymnasium as gym
import logging
import sys
import libsumo as traci
from terminal_display import terminal_display
from base_v2x_feature import BaseV2XFeature

logger = logging.getLogger("v2x.features")

#improved variable names to make their purpose more obvious and make the application be overall more readable from a 3rd person's pov
class BSMFeature(BaseV2XFeature):
    DEFAULT_LAST_BRAKE_TIME = -10.0
    MIN_BRAKE_INTERVAL_S = 0.5
    PREEMPTIVE_SLOWDOWN_FACTOR = 0.7
    
    OBSERVATION_SPACE_SIZE = 5
    ACTION_SPACE_SIZE = 1
    ACTION_DISCRETE_COUNT = 3
    ACTION_PARAMS_COUNT = 3
    
    OBSERVATION_LOW_BOUNDS = [0.0, 0.0, 0.0, 0.0, 0.0]
    OBSERVATION_HIGH_BOUNDS = [100.0, 200.0, 50.0, 10.0, 100.0]
    
    PARAM_MIN_VALUE = 0.0
    PARAM_MAX_VALUE = 1.0
    
    # Calculation constants
    MIN_RELATIVE_SPEED_THRESHOLD = 1e-3
    MAX_TIME_TO_COLLISION_CAP = 10.0
    SAFE_GAP_MULTIPLIER = 2
    SAFE_GAP_BONUS_VALUE = 0.1
    
    # Default values for initialization
    DEFAULT_MIN_GAP = 10.0
    DEFAULT_TIME_TO_COLLISION_THRESHOLD = 1.5
    DEFAULT_LEADER_DECEL_THRESHOLD = -4.0
    DEFAULT_BRAKE_DURATION = 1.0
    DEFAULT_MAX_REACT_GAP = 60.0
    DEFAULT_MAX_TIME_TO_COLLISION_GAP = 80.0
    DEFAULT_MAX_DECEL_GAP = 60.0
    
    # Range constants for RL parameters
    MIN_DETECTION_RANGE = 20.0
    MAX_DETECTION_RANGE = 100.0
    MIN_TIME_TO_COLLISION_THRESHOLD = 0.5
    MAX_TIME_TO_COLLISION_THRESHOLD = 3.0
    MIN_BRAKE_DURATION = 0.5
    MAX_BRAKE_DURATION = 2.0
    
    # Reward weight constants
    WEIGHT_TIME_TO_COLLISION = 2.0
    WEIGHT_BRAKE = 0.5
    WEIGHT_CRITICAL = 5.0
    WEIGHT_SAFE = 0.2
    SLOWDOWN_WEIGHT_FACTOR = 0.5
    def __init__(
        self,
        feature_name: str = "BSMFeature",
        enabled: bool = True,
        rl_mode: bool = False,
        min_gap: float = DEFAULT_MIN_GAP,
        time_to_collision_threshold_s: float = DEFAULT_TIME_TO_COLLISION_THRESHOLD,
        leader_decel_threshold_meters_per_second_squared: float = DEFAULT_LEADER_DECEL_THRESHOLD,
        brake_duration_s: float = DEFAULT_BRAKE_DURATION,
        max_react_gap_m: float = DEFAULT_MAX_REACT_GAP,
        max_time_to_collision_gap_m: float = DEFAULT_MAX_TIME_TO_COLLISION_GAP,
        max_decel_gap_m: float = DEFAULT_MAX_DECEL_GAP,
    ):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.rl_mode = rl_mode
        self.min_gap = float(min_gap)
        self.time_to_collision_threshold_s = float(time_to_collision_threshold_s)
        self.leader_decel_threshold_meters_per_second_squared = float(leader_decel_threshold_meters_per_second_squared)
        self.brake_duration_s = float(brake_duration_s)
        self.max_react_gap_m = float(max_react_gap_m)
        self.max_time_to_collision_gap_m = float(max_time_to_collision_gap_m)
        self.max_decel_gap_m = float(max_decel_gap_m)

        self.min_detection_range = self.MIN_DETECTION_RANGE
        self.max_detection_range = self.MAX_DETECTION_RANGE
        self.min_time_to_collision_threshold = self.MIN_TIME_TO_COLLISION_THRESHOLD
        self.max_time_to_collision_threshold = self.MAX_TIME_TO_COLLISION_THRESHOLD
        self.min_brake_duration = self.MIN_BRAKE_DURATION
        self.max_brake_duration = self.MAX_BRAKE_DURATION

        self._emergency_brake_count = 0
        self._slowdown_count = 0
        self._critical_time_to_collision_count = 0

        # Reward weights for different safety metrics
        self.weight_time_to_collision = self.WEIGHT_TIME_TO_COLLISION
        self.weight_brake = self.WEIGHT_BRAKE
        self.weight_critical = self.WEIGHT_CRITICAL
        self.weight_safe = self.WEIGHT_SAFE
        self.slowdown_weight_factor = self.SLOWDOWN_WEIGHT_FACTOR

        self.observation_size = self.OBSERVATION_SPACE_SIZE
        self.action_size = self.ACTION_SPACE_SIZE
        self._last_brake_step = {}

    def get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=np.array(self.OBSERVATION_LOW_BOUNDS, dtype=np.float32),
            high=np.array(self.OBSERVATION_HIGH_BOUNDS, dtype=np.float32),
            shape=(self.OBSERVATION_SPACE_SIZE,),
            dtype=np.float32
        )

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Dict({
            "bsm_action": gym.spaces.Discrete(self.ACTION_DISCRETE_COUNT),
            "params": gym.spaces.Box(
                low=self.PARAM_MIN_VALUE, 
                high=self.PARAM_MAX_VALUE, 
                shape=(self.ACTION_PARAMS_COUNT,), 
                dtype=np.float32
            )
        })

    def get_observation(self) -> np.ndarray:
        if not self.enable:
            return np.zeros(self.OBSERVATION_SPACE_SIZE, dtype=np.float32)
        
        vehicle_ids = traci.vehicle.getIDList()
        if len(vehicle_ids) == 0:
            return np.zeros(self.OBSERVATION_SPACE_SIZE, dtype=np.float32)
        
        # Collect metrics across all vehicle pairs
        vehicle_pair_count = 0
        total_gap = self.PARAM_MIN_VALUE
        total_rel_speed = self.PARAM_MIN_VALUE
        total_time_to_collision = self.PARAM_MIN_VALUE
        
        for vehicle_id in vehicle_ids:
            leader_data = traci.vehicle.getLeader(vehicle_id, dist=self.max_react_gap_m)
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
            
            rel_speed = max(self.PARAM_MIN_VALUE, follower_speed - leader_speed)
            time_to_collision = self._calculate_time_to_collision(gap, follower_speed, leader_speed)
            
            vehicle_pair_count += 1
            total_gap += gap
            total_rel_speed += rel_speed
            total_time_to_collision += min(time_to_collision, self.MAX_TIME_TO_COLLISION_CAP)
        
        if vehicle_pair_count > 0:
            avg_gap_distance = total_gap / vehicle_pair_count
            avg_relative_speed = total_rel_speed / vehicle_pair_count
            avg_time_to_collision = total_time_to_collision / vehicle_pair_count
        else:
            avg_gap_distance = self.PARAM_MIN_VALUE
            avg_relative_speed = self.PARAM_MIN_VALUE
            avg_time_to_collision = self.PARAM_MIN_VALUE
        
       
        logger.debug(
            f"[{self.feature_name}] Observation: pairs={vehicle_pair_count}, "
            f"avg_gap={avg_gap_distance:.2f}m, avg_rel_speed={avg_relative_speed:.2f}m/s, "
            f"avg_time_to_collision={avg_time_to_collision:.2f}s, brakes={self._emergency_brake_count}"
        )
        
        return np.array([
            float(vehicle_pair_count),
            avg_gap_distance,
            avg_relative_speed,
            avg_time_to_collision,
            float(self._emergency_brake_count)
        ], dtype=np.float32)

    def calculate_reward(self) -> float:
        if not self.enable:
            return self.PARAM_MIN_VALUE
        
        vehicle_ids = traci.vehicle.getIDList()
        if len(vehicle_ids) == 0:
            return self.PARAM_MIN_VALUE
        
        total_time_to_collision_penalty = self.PARAM_MIN_VALUE
        critical_time_to_collision_count = 0
        safe_gap_bonus = self.PARAM_MIN_VALUE
        
        # Iterate through vehicle pairs and collect time-to-collision values
        for vehicle_id in vehicle_ids:
            leader_data = traci.vehicle.getLeader(vehicle_id, dist=self.max_react_gap_m)
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
            
            # Calculate relative speed (only positive, approaching)
            rel_speed = max(self.PARAM_MIN_VALUE, follower_speed - leader_speed)
            
            # Calculate time to collision
            time_to_collision = self._calculate_time_to_collision(gap, follower_speed, leader_speed)
            
            # Penalize low time-to-collision values
            if time_to_collision < self.time_to_collision_threshold_s:
                total_time_to_collision_penalty += (self.time_to_collision_threshold_s - time_to_collision)
                critical_time_to_collision_count += 1
            
            # Reward safe following distances (consider vehicle speed)
            # For stationary or slow-moving vehicles, smaller gaps are acceptable
            # This addresses the concern that RL agents would be punished for maintaining
            # appropriate gaps at stoplights where vehicles naturally cluster closer together
            # Scale minimum gap requirement based on speed: slower vehicles can maintain smaller gaps safely
            speed_factor = max(0.3, min(1.0, follower_speed / 15.0))  # Scale from 0.3 to 1.0 based on speed (0-15 m/s)
            speed_adjusted_min_gap = self.min_gap * speed_factor
            
            if gap > speed_adjusted_min_gap * self.SAFE_GAP_MULTIPLIER:
                safe_gap_bonus += self.SAFE_GAP_BONUS_VALUE
        
        reward = (
            -self.weight_time_to_collision * total_time_to_collision_penalty
            - self.weight_brake * (self._emergency_brake_count + self._slowdown_count * self.slowdown_weight_factor)
            - self.weight_critical * critical_time_to_collision_count
            + self.weight_safe * safe_gap_bonus
        )
        
        logger.debug(
            f"[{self.feature_name}] reward: time_to_collision_penalty={total_time_to_collision_penalty:.2f}, "
            f"brakes={self._emergency_brake_count}, slowdowns={self._slowdown_count}, "
            f"critical={critical_time_to_collision_count}, safe_bonus={safe_gap_bonus:.2f}, "
            f"total={reward:.3f}"
        )
        
        return float(reward)

    def get_feature_name(self) -> str:
        return self.feature_name

    def _calculate_time_to_collision(self, gap: float, follower_speed: float, leader_speed: float) -> float:
        rel_speed = max(self.PARAM_MIN_VALUE, follower_speed - leader_speed)
        
        if rel_speed > self.MIN_RELATIVE_SPEED_THRESHOLD:
            return gap / rel_speed
        else:
            return float('inf')

    def _apply_rl_parameters(self, params):
        alpha = np.clip(float(params[0]), self.PARAM_MIN_VALUE, self.PARAM_MAX_VALUE)
        beta = np.clip(float(params[1]), self.PARAM_MIN_VALUE, self.PARAM_MAX_VALUE)
        gamma = np.clip(float(params[2]), self.PARAM_MIN_VALUE, self.PARAM_MAX_VALUE)
        
        if not (self.PARAM_MIN_VALUE <= params[0] <= self.PARAM_MAX_VALUE):
            logger.debug(f"[{self.feature_name}] Clamped params[0] from {params[0]} to {alpha}")
        if not (self.PARAM_MIN_VALUE <= params[1] <= self.PARAM_MAX_VALUE):
            logger.debug(f"[{self.feature_name}] Clamped params[1] from {params[1]} to {beta}")
        if not (self.PARAM_MIN_VALUE <= params[2] <= self.PARAM_MAX_VALUE):
            logger.debug(f"[{self.feature_name}] Clamped params[2] from {params[2]} to {gamma}")
        
        self.max_react_gap_m = (
            self.min_detection_range + 
            alpha * (self.max_detection_range - self.min_detection_range)
        )
        
        self.time_to_collision_threshold_s = (
            self.min_time_to_collision_threshold + 
            beta * (self.max_time_to_collision_threshold - self.min_time_to_collision_threshold)
        )
        
        self.brake_duration_s = (
            self.min_brake_duration + 
            gamma * (self.max_brake_duration - self.min_brake_duration)
        )
        
        logger.debug(
            f"[{self.feature_name}] RL params applied: "
            f"detection_range={self.max_react_gap_m:.2f}m, "
            f"time_to_collision_threshold={self.time_to_collision_threshold_s:.2f}s, "
            f"brake_duration={self.brake_duration_s:.2f}s"
        )

    def _get_at_risk_vehicle_pairs(self):
        at_risk_pairs = []
        current_time_s = traci.simulation.getTime()
        
        for vehicle_id in traci.vehicle.getIDList():
            leader_data = traci.vehicle.getLeader(vehicle_id, dist=self.max_react_gap_m)
            if not leader_data:
                continue

            leader_id, distance_to_leader_m = leader_data
            if leader_id is None or distance_to_leader_m < 0 or distance_to_leader_m > self.max_react_gap_m:
                continue

            try:
                follower_speed_meters_per_second = traci.vehicle.getSpeed(vehicle_id)
                leader_speed_meters_per_second = traci.vehicle.getSpeed(leader_id)
                leader_acceleration_meters_per_second_squared = traci.vehicle.getAcceleration(leader_id)
            except traci.TraCIException:
                continue

            relative_speed_meters_per_second = max(self.PARAM_MIN_VALUE, follower_speed_meters_per_second - leader_speed_meters_per_second)
            time_to_collision_s = self._calculate_time_to_collision(distance_to_leader_m, follower_speed_meters_per_second, leader_speed_meters_per_second)

            gap_trigger = (distance_to_leader_m < self.min_gap)
            time_to_collision_trigger = (
                distance_to_leader_m <= self.max_time_to_collision_gap_m
                and time_to_collision_s < self.time_to_collision_threshold_s
            )
            decel_trigger = (
                distance_to_leader_m <= self.max_decel_gap_m
                and leader_acceleration_meters_per_second_squared <= self.leader_decel_threshold_meters_per_second_squared
            )

            should_brake = gap_trigger or time_to_collision_trigger or decel_trigger

            # Check brake interval
            last_brake_time = self._last_brake_step.get(vehicle_id, self.DEFAULT_LAST_BRAKE_TIME)
            if should_brake and (current_time_s - last_brake_time) >= self.MIN_BRAKE_INTERVAL_S:
                at_risk_pairs.append((vehicle_id, leader_id, distance_to_leader_m, time_to_collision_s))
        
        return at_risk_pairs

    def _apply_rl_action(self, bsm_action):
        if not self.enable:
            return

        at_risk_pairs = self._get_at_risk_vehicle_pairs()
        
        current_time_s = traci.simulation.getTime()
        
        if bsm_action == 0:  # Emergency brake
            for follower_id, leader_id, gap, time_to_collision in at_risk_pairs:
                self._emergency_brake_count += 1
                try:
                    traci.vehicle.slowDown(follower_id, self.PARAM_MIN_VALUE, self.brake_duration_s)
                    self._last_brake_step[follower_id] = current_time_s
                    logger.debug(
                        f"[{self.feature_name}] RL EMERGENCY_BRAKE: {follower_id} -> {leader_id} "
                        f"(gap={gap:.1f}m, time_to_collision={time_to_collision:.2f}s)"
                    )
                except traci.TraCIException as e:
                    logger.warning(f"[{self.feature_name}] Emergency brake failed for {follower_id}: {e}")
        
        elif bsm_action == 1:  # Preemptive slowdown
            for follower_id, leader_id, gap, time_to_collision in at_risk_pairs:
                self._slowdown_count += 1
                try:
                    current_speed = traci.vehicle.getSpeed(follower_id)
                    target_speed = max(self.PARAM_MIN_VALUE, current_speed * self.PREEMPTIVE_SLOWDOWN_FACTOR)
                    traci.vehicle.slowDown(follower_id, target_speed, self.brake_duration_s)
                    self._last_brake_step[follower_id] = current_time_s
                    logger.debug(
                        f"[{self.feature_name}] RL PREEMPTIVE_SLOWDOWN: {follower_id} -> {leader_id} "
                        f"(gap={gap:.1f}m, time_to_collision={time_to_collision:.2f}s, target={target_speed:.2f}m/s)"
                    )
                except traci.TraCIException as e:
                    logger.warning(f"[{self.feature_name}] Preemptive slowdown failed for {follower_id}: {e}")
        
        elif bsm_action == 2:  # No intervention
            logger.debug(f"[{self.feature_name}] RL NO_INTERVENTION: {len(at_risk_pairs)} at-risk pairs")
            pass

    def _log_bsm_events(self, events: dict):
        any_events = any(len(v) for v in events.values())
        if not any_events:
            return

        summary_parts = []
        em_count = len(events["EMERGENCY_BRAKE"])
        pre_count = len(events["PREEMPTIVE_SLOWDOWN"])
        warn_count = len(events["WARN"])
        if em_count:
            summary_parts.append(f"EMG={em_count}")
        if pre_count:
            summary_parts.append(f"PRE={pre_count}")
        if warn_count:
            summary_parts.append(f"WARN={warn_count}")

        latest_short = None
        if events["EMERGENCY_BRAKE"]:
            latest_short = events["EMERGENCY_BRAKE"][-1][1]
        elif events["PREEMPTIVE_SLOWDOWN"]:
            latest_short = events["PREEMPTIVE_SLOWDOWN"][-1][1]
        elif events["WARN"]:
            latest_short = events["WARN"][-1][1]

        summary = f"[{self.feature_name}] | " + " ".join(summary_parts)
        if latest_short:
            summary += f" | {latest_short}"

        terminal_display.update("BSM", summary)
        terminal_display.render()

        if not sys.stdout.isatty():
            for typ in ("EMERGENCY_BRAKE", "PREEMPTIVE_SLOWDOWN", "WARN"):
                for verbose, _ in events[typ]:
                    logger.info(verbose)

    def take_action(self, action):
        """Main action method that delegates to RL or rule-based implementation"""
        if not self.enable:
            return

        self._emergency_brake_count = 0
        self._slowdown_count = 0
        self._critical_time_to_collision_count = 0

        if self.rl_mode:
            self._take_rl_action(action)
        else:
            self._take_rule_based_action()

    def _take_rl_action(self, action):
        """Handle RL actions with dictionary format"""
        try:
            # Validate dictionary structure
            if not isinstance(action, dict) or "bsm_action" not in action or "params" not in action:
                logger.warning(
                    f"[{self.feature_name}] Invalid RL action format, falling back to rule-based mode"
                )
                self._take_rule_based_action()
                return

            bsm_action = action["bsm_action"]
            params = action["params"]
            
            logger.debug(
                f"[{self.feature_name}] RL mode: action={bsm_action}, params={params}"
            )

            self._apply_rl_parameters(params)
            self._apply_rl_action(bsm_action)
            
        except Exception as e:
            logger.warning(
                f"[{self.feature_name}] RL action execution failed: {e}, falling back to rule-based mode"
            )
            self._take_rule_based_action()

    def _take_rule_based_action(self):
        """Handle rule-based actions using predefined logic"""
        current_time_s = traci.simulation.getTime()
        
        events = {
            "EMERGENCY_BRAKE": [],
            "PREEMPTIVE_SLOWDOWN": [],
            "WARN": [],
        }

        for vehicle_id in traci.vehicle.getIDList():
            leader_data = traci.vehicle.getLeader(vehicle_id, dist=self.max_react_gap_m)
            if not leader_data:
                continue

            leader_id, distance_to_leader_m = leader_data
            if leader_id is None or distance_to_leader_m < 0 or distance_to_leader_m > self.max_react_gap_m:
                continue

            try:
                follower_speed_meters_per_second = traci.vehicle.getSpeed(vehicle_id)
                leader_speed_meters_per_second = traci.vehicle.getSpeed(leader_id)
                leader_acceleration_meters_per_second_squared = traci.vehicle.getAcceleration(leader_id)
            except traci.TraCIException:
                continue

            relative_speed_meters_per_second = max(self.PARAM_MIN_VALUE, follower_speed_meters_per_second - leader_speed_meters_per_second)
            time_to_collision_s = self._calculate_time_to_collision(distance_to_leader_m, follower_speed_meters_per_second, leader_speed_meters_per_second)

            gap_trigger = (distance_to_leader_m < self.min_gap)
            time_to_collision_trigger = (
                distance_to_leader_m <= self.max_time_to_collision_gap_m
                and time_to_collision_s < self.time_to_collision_threshold_s
            )
            decel_trigger = (
                distance_to_leader_m <= self.max_decel_gap_m
                and leader_acceleration_meters_per_second_squared <= self.leader_decel_threshold_meters_per_second_squared
            )

            should_brake = gap_trigger or time_to_collision_trigger or decel_trigger

            # this is the part that collects reasons for the logging
            trigger_reasons = []
            if gap_trigger:
                trigger_reasons.append(f"GAP<{self.min_gap:.1f}m")
            if time_to_collision_trigger:
                trigger_reasons.append(
                    f"TIME_TO_COLLISION<{self.time_to_collision_threshold_s:.2f}s"
                )
            if decel_trigger:
                trigger_reasons.append(
                    f"LEADER_DECEL<={self.leader_decel_threshold_meters_per_second_squared:.1f}m/s²"
                )

            last_brake_time = self._last_brake_step.get(vehicle_id, self.DEFAULT_LAST_BRAKE_TIME)
            if should_brake and (current_time_s - last_brake_time) >= self.MIN_BRAKE_INTERVAL_S:

                time_to_collision_display = (
                    f"{time_to_collision_s:.2f}s"
                    if np.isfinite(time_to_collision_s)
                    else "distance growing"
                )

                if gap_trigger or time_to_collision_trigger:
                    verbose = (
                        f"[{self.feature_name}] EMERGENCY_BRAKE: {vehicle_id} -> leader {leader_id} "
                        f"(gap={distance_to_leader_m:.1f}m, time_to_collision={time_to_collision_display}, "
                        f"reasons=[{', '.join(trigger_reasons)}]) @ {current_time_s:.1f}s"
                    )
                    # short form for TTY summary
                    short = f"EMG:{vehicle_id}->{leader_id} gap={distance_to_leader_m:.1f}m"
                    events["EMERGENCY_BRAKE"].append((verbose, short))
                    try:
                        traci.vehicle.slowDown(vehicle_id, self.PARAM_MIN_VALUE, self.brake_duration_s)
                    except traci.TraCIException as e:
                        warn = f"[{self.feature_name}] brake failed for {vehicle_id}: {e}"
                        events["WARN"].append((warn, warn))
                        logger.warning(warn)
                else:
                    target_speed_meters_per_second = max(self.PARAM_MIN_VALUE, follower_speed_meters_per_second * self.PREEMPTIVE_SLOWDOWN_FACTOR)
                    verbose = (
                        f"[{self.feature_name}] PREEMPTIVE_SLOWDOWN: {vehicle_id} following {leader_id} "
                        f"(gap={distance_to_leader_m:.1f}m, time_to_collision={time_to_collision_display}, "
                        f"target={target_speed_meters_per_second:.2f}m/s, reasons=[{', '.join(trigger_reasons)}]) @ {current_time_s:.1f}s"
                    )
                    short = f"PRE:{vehicle_id}->{leader_id} gap={distance_to_leader_m:.1f}m tgt={target_speed_meters_per_second:.1f}m/s"
                    events["PREEMPTIVE_SLOWDOWN"].append((verbose, short))
                    try:
                        traci.vehicle.slowDown(vehicle_id, target_speed_meters_per_second, self.brake_duration_s)
                    except traci.TraCIException as e:
                        warn = f"[{self.feature_name}] slowdown failed for {vehicle_id}: {e}"
                        events["WARN"].append((warn, warn))
                        logger.warning(warn)

                self._last_brake_step[vehicle_id] = current_time_s

        # Emit aggregated output
        self._log_bsm_events(events)

    def feature_reset(self):
        self._last_brake_step.clear()
        self._emergency_brake_count = 0
        self._slowdown_count = 0
        self._critical_time_to_collision_count = 0
        logger.debug(f"[{self.feature_name}] Feature reset: cleared brake history and RL metrics")

    # the distance growing is there for when theh gap between cars is rather small but currently growing so there's no risk of collision

    def _trigger_emergency_brake(self, vehicle_id: str, gap: float, time_to_collision_s: float, leader_id: str, simulation_time: float):
        time_to_collision_display = (
            f"{time_to_collision_s:.2f}s" if np.isfinite(time_to_collision_s) else "distance growing"
        )
        logger.info(
            f"[{self.feature_name}] BSM: {vehicle_id} EMERGENCY_BRAKE "
            f"(leader={leader_id}, gap={gap:.1f}m, time_to_collision={time_to_collision_display}) @ {simulation_time:.1f}s"
        )
        try:
            traci.vehicle.slowDown(vehicle_id, self.PARAM_MIN_VALUE, self.brake_duration_s)
        except traci.TraCIException as e:
            logger.warning(f"[{self.feature_name}] brake failed for {vehicle_id}: {e}")