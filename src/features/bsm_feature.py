import numpy as np
import gymnasium as gym
import logging
import sys
import libsumo as traci
from ui.terminal_display import terminal_display
from base_v2x_feature import BaseV2XFeature

logger = logging.getLogger("v2x.features")

class BSMFeature(BaseV2XFeature):
    DEFAULT_LAST_BRAKE_TIME = -10.0
    MIN_BRAKE_INTERVAL_S = 0.5
    PREEMPTIVE_SLOWDOWN_FACTOR = 0.7

    RL_ACTION_DECREASE_TTC = 0
    RL_ACTION_INCREASE_TTC = 1
    RL_ACTION_DECREASE_GAP = 2
    RL_ACTION_INCREASE_GAP = 3
    RL_ACTION_NOOP = 4

    MIN_TTC_THRESHOLD = 0.5
    MAX_TTC_THRESHOLD = 4.0
    MIN_REACT_GAP = 20.0
    MAX_REACT_GAP = 150.0
    TTC_STEP_SIZE = 0.25
    GAP_STEP_SIZE = 5.0

    WEIGHT_WAIT = 0.1
    WEIGHT_TTC = 0.5
    WEIGHT_BRAKE = 2.0
    WEIGHT_CRITICAL = 5.0
    REWARD_NORMALIZATION = 100.0

    OBSERVATION_SIZE = 12
    ACTION_SIZE = 5

    ANGLE_NORTH_START = 315
    ANGLE_NORTH_END = 45
    ANGLE_EAST_START = 45
    ANGLE_EAST_END = 135
    ANGLE_SOUTH_START = 135
    ANGLE_SOUTH_END = 225

    NORM_MAX_VEHICLES = 20.0
    NORM_MAX_SPEED = 20.0
    NORM_MAX_WAIT = 100.0

    TTC_PENALTY_OFFSET = 1.0
    TTC_CRITICAL_THRESHOLD = 0.5
    TTC_INFINITE_SPEED_THRESHOLD = 1e-3
    PREEMPTIVE_PENALTY_FACTOR = 0.5

    def __init__(
        self,
        feature_name: str = "BSMFeature",
        enabled: bool = True,
        rl_mode: bool = False,
        min_gap: float = 10.0,
        time_to_collision_threshold_s: float = 1.5,
        leader_decel_threshold_mps2: float = -4.0,
        brake_duration_s: float = 1.0,
        max_react_gap_m: float = 60.0,
        max_time_to_collision_gap_m: float = 80.0,
        max_decel_gap_m: float = 60.0,
    ):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.rl_mode = rl_mode
        self.min_gap = float(min_gap)
        self.time_to_collision_threshold_s = float(time_to_collision_threshold_s)
        self.leader_decel_threshold_mps2 = float(leader_decel_threshold_mps2)
        self.brake_duration_s = float(brake_duration_s)
        self.max_react_gap_m = float(max_react_gap_m)
        self.max_time_to_collision_gap_m = float(max_time_to_collision_gap_m)
        self.max_decel_gap_m = float(max_decel_gap_m)

        self._last_brake_step = {}
        
        self._step_stats = {
            "emergency_brake_count": 0,
            "preemptive_slowdown_count": 0,
            "total_ttc_penalty": 0.0,
            "critical_event_count": 0,
            "total_waiting_time": 0.0
        }

    def get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.OBSERVATION_SIZE,), dtype=np.float32)

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(self.ACTION_SIZE)

    def get_observation(self) -> np.ndarray:
        self._step_stats["total_waiting_time"] = 0.0
        
        approach_stats = [[0, 0.0, 0.0] for _ in range(4)]
        
        vehicle_list = traci.vehicle.getIDList()
        
        for vid in vehicle_list:
            speed = traci.vehicle.getSpeed(vid)
            angle = traci.vehicle.getAngle(vid)
            wait = traci.vehicle.getWaitingTime(vid)
                
            self._step_stats["total_waiting_time"] += wait
                
            if self.ANGLE_NORTH_START <= angle or angle < self.ANGLE_NORTH_END:
                bucket = 0
            elif self.ANGLE_EAST_START <= angle < self.ANGLE_EAST_END:
                bucket = 1
            elif self.ANGLE_SOUTH_START <= angle < self.ANGLE_SOUTH_END:
                bucket = 2
            else:
                bucket = 3
                    
            approach_stats[bucket][0] += 1
            approach_stats[bucket][1] += speed
            approach_stats[bucket][2] += wait

        obs = []
        for stats in approach_stats:
            count, s_speed, s_wait = stats
            avg_speed = (s_speed / count) if count > 0 else self.NORM_MAX_SPEED
            avg_wait = (s_wait / count) if count > 0 else 0.0
            
            obs.append(min(1.0, count / self.NORM_MAX_VEHICLES))
            obs.append(min(1.0, avg_speed / self.NORM_MAX_SPEED))
            obs.append(min(1.0, avg_wait / self.NORM_MAX_WAIT))
            
        return np.array(obs, dtype=np.float32)

    def calculate_reward(self) -> float:
        reward = 0.0
        
        reward -= self.WEIGHT_WAIT * (self._step_stats["total_waiting_time"] / self.REWARD_NORMALIZATION)
        
        reward -= self.WEIGHT_TTC * self._step_stats["total_ttc_penalty"]
        
        reward -= self.WEIGHT_BRAKE * (self._step_stats["emergency_brake_count"] + self._step_stats["preemptive_slowdown_count"] * self.PREEMPTIVE_PENALTY_FACTOR)
        
        reward -= self.WEIGHT_CRITICAL * self._step_stats["critical_event_count"]
        
        return float(reward)

    def get_feature_name(self) -> str:
        return self.feature_name

    def _log_bsm_events(self, events: dict):
        any_events = any(len(v) for v in events.values())
        if not any_events:
            if self.rl_mode:
                status = f"TTC_Thresh={self.time_to_collision_threshold_s:.2f}s | Gap={self.max_react_gap_m:.1f}m"
                terminal_display.update("BSM", f"[{self.feature_name}] {status}")
                terminal_display.render()
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
        if self.rl_mode:
             summary += f" | T:{self.time_to_collision_threshold_s:.2f}s"

        if latest_short:
            summary += f" | {latest_short}"

        terminal_display.update("BSM", summary)
        terminal_display.render()

        if not sys.stdout.isatty():
            for typ in ("EMERGENCY_BRAKE", "PREEMPTIVE_SLOWDOWN", "WARN"):
                for verbose, _ in events[typ]:
                    logger.info(verbose)

    def take_action(self, action):
        if not self.enable:
            return

        if self.rl_mode and action is not None:
             if action == self.RL_ACTION_DECREASE_TTC:
                 self.time_to_collision_threshold_s = max(self.MIN_TTC_THRESHOLD, self.time_to_collision_threshold_s - self.TTC_STEP_SIZE)
             elif action == self.RL_ACTION_INCREASE_TTC:
                 self.time_to_collision_threshold_s = min(self.MAX_TTC_THRESHOLD, self.time_to_collision_threshold_s + self.TTC_STEP_SIZE)
             elif action == self.RL_ACTION_DECREASE_GAP:
                 self.max_react_gap_m = max(self.MIN_REACT_GAP, self.max_react_gap_m - self.GAP_STEP_SIZE)
             elif action == self.RL_ACTION_INCREASE_GAP:
                 self.max_react_gap_m = min(self.MAX_REACT_GAP, self.max_react_gap_m + self.GAP_STEP_SIZE)

        self._step_stats["emergency_brake_count"] = 0
        self._step_stats["preemptive_slowdown_count"] = 0
        self._step_stats["total_ttc_penalty"] = 0.0
        self._step_stats["critical_event_count"] = 0

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

            follower_speed_mps = traci.vehicle.getSpeed(vehicle_id)
            leader_speed_mps = traci.vehicle.getSpeed(leader_id)
            leader_accel_mps2 = traci.vehicle.getAcceleration(leader_id)

            relative_speed_mps = max(0.0, follower_speed_mps - leader_speed_mps)
            time_to_collision_s = (
                distance_to_leader_m / relative_speed_mps if relative_speed_mps > self.TTC_INFINITE_SPEED_THRESHOLD else float("inf")
            )

            if time_to_collision_s < self.time_to_collision_threshold_s + self.TTC_PENALTY_OFFSET:
                 self._step_stats["total_ttc_penalty"] += (self.time_to_collision_threshold_s + self.TTC_PENALTY_OFFSET - time_to_collision_s)

            if time_to_collision_s < self.TTC_CRITICAL_THRESHOLD:
                self._step_stats["critical_event_count"] += 1

            gap_trigger = (distance_to_leader_m < self.min_gap)
            time_to_collision_trigger = (
                distance_to_leader_m <= self.max_time_to_collision_gap_m
                and time_to_collision_s < self.time_to_collision_threshold_s
            )
            decel_trigger = (
                distance_to_leader_m <= self.max_decel_gap_m
                and leader_accel_mps2 <= self.leader_decel_threshold_mps2
            )

            should_brake = gap_trigger or time_to_collision_trigger or decel_trigger

            trigger_reasons = []
            if gap_trigger:
                trigger_reasons.append(f"GAP<{self.min_gap:.1f}m")
            if time_to_collision_trigger:
                trigger_reasons.append(
                    f"TIME_TO_COLLISION<{self.time_to_collision_threshold_s:.2f}s"
                )
            if decel_trigger:
                trigger_reasons.append(
                    f"LEADER_DECEL<={self.leader_decel_threshold_mps2:.1f}m/s²"
                )

            last_brake_time = self._last_brake_step.get(vehicle_id, self.DEFAULT_LAST_BRAKE_TIME)
            if should_brake and (current_time_s - last_brake_time) >= self.MIN_BRAKE_INTERVAL_S:

                time_to_collision_display = (
                    f"{time_to_collision_s:.2f}s"
                    if np.isfinite(time_to_collision_s)
                    else "distance growing"
                )

                if gap_trigger or time_to_collision_trigger:
                    self._step_stats["emergency_brake_count"] += 1
                    
                    verbose = (
                        f"[{self.feature_name}] EMERGENCY_BRAKE: {vehicle_id} -> leader {leader_id} "
                        f"(gap={distance_to_leader_m:.1f}m, time_to_collision={time_to_collision_display}, "
                        f"reasons=[{', '.join(trigger_reasons)}]) @ {current_time_s:.1f}s"
                    )
                    short = f"EMG:{vehicle_id}->{leader_id} gap={distance_to_leader_m:.1f}m"
                    events["EMERGENCY_BRAKE"].append((verbose, short))
                    try:
                        traci.vehicle.slowDown(vehicle_id, 0.0, self.brake_duration_s)
                    except traci.TraCIException as e:
                        warn = f"[{self.feature_name}] brake failed for {vehicle_id}: {e}"
                        events["WARN"].append((warn, warn))
                        logger.warning(warn)
                else:
                    self._step_stats["preemptive_slowdown_count"] += 1
                    
                    target_speed_mps = max(0.0, follower_speed_mps * self.PREEMPTIVE_SLOWDOWN_FACTOR)
                    verbose = (
                        f"[{self.feature_name}] PREEMPTIVE_SLOWDOWN: {vehicle_id} following {leader_id} "
                        f"(gap={distance_to_leader_m:.1f}m, time_to_collision={time_to_collision_display}, "
                        f"target={target_speed_mps:.2f}m/s, reasons=[{', '.join(trigger_reasons)}]) @ {current_time_s:.1f}s"
                    )
                    short = f"PRE:{vehicle_id}->{leader_id} gap={distance_to_leader_m:.1f}m tgt={target_speed_mps:.1f}m/s"
                    events["PREEMPTIVE_SLOWDOWN"].append((verbose, short))
                    try:
                        traci.vehicle.slowDown(vehicle_id, target_speed_mps, self.brake_duration_s)
                    except traci.TraCIException as e:
                        warn = f"[{self.feature_name}] slowdown failed for {vehicle_id}: {e}"
                        events["WARN"].append((warn, warn))
                        logger.warning(warn)

                self._last_brake_step[vehicle_id] = current_time_s

        self._log_bsm_events(events)

    def feature_reset(self):
        self._last_brake_step.clear()
        self._step_stats = {
            "emergency_brake_count": 0,
            "preemptive_slowdown_count": 0,
            "total_ttc_penalty": 0.0,
            "critical_event_count": 0,
            "total_waiting_time": 0.0
        }

    def _trigger_emergency_brake(self, veh_id: str, gap: float, time_to_collision_s: float, leader_id: str, sim_t: float):
        time_to_collision_display = (
            f"{time_to_collision_s:.2f}s" if np.isfinite(time_to_collision_s) else "distance growing"
        )
        logger.info(
            f"[{self.feature_name}] BSM: {veh_id} EMERGENCY_BRAKE "
            f"(leader={leader_id}, gap={gap:.1f}m, time_to_collision={time_to_collision_display}) @ {sim_t:.1f}s"
        )
        try:
            traci.vehicle.slowDown(veh_id, 0.0, self.brake_duration_s)
        except traci.TraCIException as e:
            logger.warning(f"[{self.feature_name}] brake failed for {veh_id}: {e}")

