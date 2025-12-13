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
    def __init__(
        self,
        feature_name: str = "BSMFeature",
        enabled: bool = True,
        min_gap: float = 10.0,
        time_to_collision_threshold_s: float = 1.5,
        leader_decel_threshold_meters_per_second_squared: float = -4.0,
        brake_duration_s: float = 1.0,
        max_react_gap_m: float = 60.0,
        max_time_to_collision_gap_m: float = 80.0,
        max_decel_gap_m: float = 60.0,
    ):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.min_gap = float(min_gap)
        self.time_to_collision_threshold_s = float(time_to_collision_threshold_s)
        self.leader_decel_threshold_meters_per_second_squared = float(leader_decel_threshold_meters_per_second_squared)
        self.brake_duration_s = float(brake_duration_s)
        self.max_react_gap_m = float(max_react_gap_m)
        self.max_time_to_collision_gap_m = float(max_time_to_collision_gap_m)
        self.max_decel_gap_m = float(max_decel_gap_m)

        self.min_detection_range = 20.0
        self.max_detection_range = 100.0
        self.min_time_to_collision_threshold = 0.5
        self.max_time_to_collision_threshold = 3.0
        self.min_brake_duration = 0.5
        self.max_brake_duration = 2.0

        self._emergency_brake_count = 0
        self._slowdown_count = 0
        self._critical_time_to_collision_count = 0

        # Reward weights for different safety metrics
        self.weight_time_to_collision = 2.0        
        self.weight_brake = 0.5      
        self.weight_critical = 5.0   
        self.weight_safe = 0.2       
        self.slowdown_weight_factor = 0.5  

        self.observation_size = 5
        self.action_size = 1
        self._last_brake_step = {}

    def get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 200.0, 50.0, 10.0, 100.0], dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Dict({
            "bsm_action": gym.spaces.Discrete(3),
            "params": gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        })

    def get_observation(self) -> np.ndarray:
        if not self.enable:
            return np.zeros(5, dtype=np.float32)
        
        vehicle_ids = traci.vehicle.getIDList()
        if len(vehicle_ids) == 0:
            return np.zeros(5, dtype=np.float32)
        
        # Collect metrics across all vehicle pairs
        vehicle_pair_count = 0
        total_gap = 0.0
        total_rel_speed = 0.0
        total_time_to_collision = 0.0
        
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
            
            rel_speed = max(0.0, follower_speed - leader_speed)
            
            if rel_speed > 1e-3:
                time_to_collision = gap / rel_speed
            else:
                time_to_collision = float('inf')
            
            vehicle_pair_count += 1
            total_gap += gap
            total_rel_speed += rel_speed
            total_time_to_collision += min(time_to_collision, 10.0)
        
        if vehicle_pair_count > 0:
            avg_gap_distance = total_gap / vehicle_pair_count
            avg_relative_speed = total_rel_speed / vehicle_pair_count
            avg_time_to_collision = total_time_to_collision / vehicle_pair_count
        else:
            avg_gap_distance = 0.0
            avg_relative_speed = 0.0
            avg_time_to_collision = 0.0
        
       
        logger.debug(
            f"[{self.feature_name}] Observation: pairs={vehicle_pair_count}, "
            f"avg_gap={avg_gap_distance:.2f}m, avg_rel_speed={avg_relative_speed:.2f}m/s, "
            f"avg_time_to_collision={avg_time_to_collision:.2f}s, brakes={self._emergency_brake_count}"
        )
        
         # Return observation array with shape (5,) and dtype float32
        return np.array([
            float(vehicle_pair_count),
            avg_gap_distance,
            avg_relative_speed,
            avg_time_to_collision,
            float(self._emergency_brake_count)
        ], dtype=np.float32)

    def calculate_reward(self) -> float:
        if not self.enable:
            return 0.0
        
        vehicle_ids = traci.vehicle.getIDList()
        if len(vehicle_ids) == 0:
            return 0.0
        
        total_time_to_collision_penalty = 0.0
        critical_time_to_collision_count = 0
        safe_gap_bonus = 0.0
        
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
            rel_speed = max(0.0, follower_speed - leader_speed)
            
            # Calculate time to collision
            if rel_speed > 1e-3:
                time_to_collision = gap / rel_speed
            else:
                time_to_collision = float('inf')
            
            # Penalize low time-to-collision values
            if time_to_collision < self.time_to_collision_threshold_s:
                total_time_to_collision_penalty += (self.time_to_collision_threshold_s - time_to_collision)
                critical_time_to_collision_count += 1
            
            # Reward safe following distances
            if gap > self.min_gap * 2:
                safe_gap_bonus += 0.1
        
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

    def _apply_rl_parameters(self, params):
        alpha = np.clip(float(params[0]), 0.0, 1.0)
        beta = np.clip(float(params[1]), 0.0, 1.0)
        gamma = np.clip(float(params[2]), 0.0, 1.0)
        
        if not (0.0 <= params[0] <= 1.0):
            logger.debug(f"[{self.feature_name}] Clamped params[0] from {params[0]} to {alpha}")
        if not (0.0 <= params[1] <= 1.0):
            logger.debug(f"[{self.feature_name}] Clamped params[1] from {params[1]} to {beta}")
        if not (0.0 <= params[2] <= 1.0):
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

            relative_speed_meters_per_second = max(0.0, follower_speed_meters_per_second - leader_speed_meters_per_second)
            time_to_collision_s = (
                distance_to_leader_m / relative_speed_meters_per_second if relative_speed_meters_per_second > 1e-3 else float("inf")
            )

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
                    traci.vehicle.slowDown(follower_id, 0.0, self.brake_duration_s)
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
                    target_speed = max(0.0, current_speed * self.PREEMPTIVE_SLOWDOWN_FACTOR)
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

    # O(n), utilizes get leader function from TRACI api, if there is no leader ahead of a car then it skips the checks for it
    # changes were made to differentiate between situations that trigger any sort of slowdown so that the logs are clearer
    def take_action(self, action):
        if not self.enable:
            return

        self._emergency_brake_count = 0
        self._slowdown_count = 0
        self._critical_time_to_collision_count = 0

        # Mode detection: check if action is a dictionary (RL mode) or not (rule-based mode)
        rl_mode = False
        try:
            rl_mode = isinstance(action, dict)
            if rl_mode:
                # Validate dictionary structure
                if "bsm_action" not in action or "params" not in action:
                    logger.warning(
                        f"[{self.feature_name}] Invalid action dict (missing keys), using rule-based mode"
                    )
                    rl_mode = False
        except Exception as e:
            logger.warning(
                f"[{self.feature_name}] Action processing error: {e}, using rule-based mode"
            )
            rl_mode = False

        if rl_mode:
            try:
                bsm_action = action["bsm_action"]
                params = action["params"]
                
                logger.debug(
                    f"[{self.feature_name}] RL mode: action={bsm_action}, params={params}"
                )

                self._apply_rl_parameters(params)
                self._apply_rl_action(bsm_action)
                return
            except Exception as e:
                logger.warning(
                    f"[{self.feature_name}] RL action execution failed: {e}, falling back to rule-based mode"
                )
                
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

            relative_speed_meters_per_second = max(0.0, follower_speed_meters_per_second - leader_speed_meters_per_second)
            time_to_collision_s = (
                distance_to_leader_m / relative_speed_meters_per_second if relative_speed_meters_per_second > 1e-3 else float("inf")
            )


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
                        traci.vehicle.slowDown(vehicle_id, 0.0, self.brake_duration_s)
                    except traci.TraCIException as e:
                        warn = f"[{self.feature_name}] brake failed for {vehicle_id}: {e}"
                        events["WARN"].append((warn, warn))
                        logger.warning(warn)
                else:
                    target_speed_meters_per_second = max(0.0, follower_speed_meters_per_second * self.PREEMPTIVE_SLOWDOWN_FACTOR)
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
            traci.vehicle.slowDown(vehicle_id, 0.0, self.brake_duration_s)
        except traci.TraCIException as e:
            logger.warning(f"[{self.feature_name}] brake failed for {vehicle_id}: {e}")