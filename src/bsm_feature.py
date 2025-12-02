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
        leader_decel_threshold_mps2: float = -4.0,
        brake_duration_s: float = 1.0,
        max_react_gap_m: float = 60.0,
        max_time_to_collision_gap_m: float = 80.0,
        max_decel_gap_m: float = 60.0,
        reward_weight_waiting: float = 0.7,
        reward_weight_flow: float = 0.3,
    ):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.min_gap = float(min_gap)
        self.time_to_collision_threshold_s = float(time_to_collision_threshold_s)
        self.leader_decel_threshold_mps2 = float(leader_decel_threshold_mps2)
        self.brake_duration_s = float(brake_duration_s)
        self.max_react_gap_m = float(max_react_gap_m)
        self.max_time_to_collision_gap_m = float(max_time_to_collision_gap_m)
        self.max_decel_gap_m = float(max_decel_gap_m)

        # RL tracking attributes
        self._previous_waiting_time = 0.0
        self._vehicles_passed_count = 0
        self._last_vehicle_set = set()
        
        # Reward configuration parameters
        self._reward_weight_waiting = float(reward_weight_waiting)
        self._reward_weight_flow = float(reward_weight_flow)
        
        # Normalization constants
        self.max_vehicles_per_approach = 50
        self.max_speed = 50.0  # m/s (180 km/h)
        self.max_waiting_time = 300.0  # seconds
        self.max_total_vehicles = 200

        self.observation_size = 10
        self.action_size = 1
        self._last_brake_step = {}

    def get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_size,))

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(self.action_size)

    def get_observation(self) -> np.ndarray:
        try:
            positions = self._extract_vehicle_positions()
            speeds = self._extract_vehicle_speeds()
            directions = self._extract_vehicle_directions()
            
            
            vehicle_counts = self._compute_vehicle_counts_per_approach(positions, directions)
            avg_speeds = self._compute_average_speeds_per_approach(speeds, directions)
            global_waiting_time = self._compute_global_waiting_time()
            total_vehicles = len(positions)
            
           
            normalized_counts = np.array([
                self._normalize_observation_component(vehicle_counts[i], self.max_vehicles_per_approach)
                for i in range(4)
            ], dtype=np.float32)
            
            normalized_speeds = np.array([
                self._normalize_observation_component(avg_speeds[i], self.max_speed)
                for i in range(4)
            ], dtype=np.float32)
            
           
            avg_waiting_time = global_waiting_time / total_vehicles if total_vehicles > 0 else 0.0
            normalized_waiting = self._normalize_observation_component(avg_waiting_time, self.max_waiting_time)
            
            normalized_total_vehicles = self._normalize_observation_component(total_vehicles, self.max_total_vehicles)
            
           
            observation = np.concatenate([
                normalized_counts,      
                normalized_speeds,      
                [normalized_waiting],  
                [normalized_total_vehicles]  
            ]).astype(np.float32)
            
            return observation
            
        except Exception as e:
            logger.warning(f"Failed to extract observation: {e}")
            return np.zeros(self.observation_size, dtype=np.float32)

    def calculate_reward(self) -> float:
        try:
            waiting_reward = self._calculate_waiting_time_reward()
            flow_reward = self._calculate_flow_reward()
            
            
            total_reward = (
                self._reward_weight_waiting * waiting_reward +
                self._reward_weight_flow * flow_reward
            )
            
            return total_reward
            
        except Exception as e:
            logger.warning(f"Failed to calculate reward: {e}")
            return 0.0
    
    def _calculate_flow_reward(self) -> float:
        try:
            current_vehicle_set = set(traci.vehicle.getIDList())
            
            vehicles_completed = self._last_vehicle_set - current_vehicle_set
            num_completed = len(vehicles_completed)
            
            vehicles_departed = current_vehicle_set - self._last_vehicle_set
            num_departed = len(vehicles_departed)
            
            self._last_vehicle_set = current_vehicle_set
            
            expected_arrival_rate = 5.0
            
            flow_reward = num_completed / expected_arrival_rate
            
            return flow_reward
            
        except Exception as e:
            logger.warning(f"Failed to calculate flow reward: {e}")
            return 0.0
    
    def _calculate_waiting_time_reward(self) -> float:
        try:
            current_total_waiting = 0.0
            vehicle_ids = traci.vehicle.getIDList()
            num_vehicles = len(vehicle_ids)
            
            for vehicle_id in vehicle_ids:
                try:
                    waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                    current_total_waiting += waiting_time
                except Exception as e:
                    logger.warning(f"Failed to get waiting time for {vehicle_id}: {e}")
                    continue
            
            delta_waiting = current_total_waiting - self._previous_waiting_time
            
            if num_vehicles > 0:
                normalized_delta = delta_waiting / num_vehicles
            else:
                normalized_delta = 0.0
            
            reward = -normalized_delta
            
            self._previous_waiting_time = current_total_waiting
            
            return reward
            
        except Exception as e:
            logger.warning(f"Failed to calculate waiting time reward: {e}")
            return 0.0

    def get_feature_name(self) -> str:
        return self.feature_name

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
        if not self.enable:
            return

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
                follower_speed_mps = traci.vehicle.getSpeed(vehicle_id)
                leader_speed_mps = traci.vehicle.getSpeed(leader_id)
                leader_accel_mps2 = traci.vehicle.getAcceleration(leader_id)
            except traci.TraCIException:
                continue

            relative_speed_mps = max(0.0, follower_speed_mps - leader_speed_mps)
            time_to_collision_s = (
                distance_to_leader_m / relative_speed_mps if relative_speed_mps > 1e-3 else float("inf")
            )


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

    def _extract_vehicle_positions(self) -> dict:
        positions = {}
        try:
            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                try:
                    position = traci.vehicle.getPosition(vehicle_id)
                    positions[vehicle_id] = position
                except Exception as e:
                    logger.warning(f"Failed to get position for {vehicle_id}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Failed to get vehicle ID list: {e}")
        
        return positions
    
    def _extract_vehicle_speeds(self) -> dict:
        speeds = {}
        try:
            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                try:
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    speeds[vehicle_id] = speed
                except Exception as e:
                    logger.warning(f"Failed to get speed for {vehicle_id}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Failed to get vehicle ID list: {e}")
        
        return speeds
    
    def _extract_vehicle_directions(self) -> dict:   
        directions = {}
        try:
            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                try:
                    angle = traci.vehicle.getAngle(vehicle_id)
                    directions[vehicle_id] = angle
                except Exception as e:
                    logger.warning(f"Failed to get angle for {vehicle_id}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Failed to get vehicle ID list: {e}")
        
        return directions

    def _compute_vehicle_counts_per_approach(self, positions: dict, directions: dict) -> np.ndarray:  
        counts = np.zeros(4, dtype=np.float32)  # [N, S, E, W]
        
        for vehicle_id in positions.keys():
            if vehicle_id not in directions:
                continue
                
            angle = directions[vehicle_id]
            
            # Map angle to approach direction
            # North: 315-45 degrees (0 degrees is East in SUMO)
            # East: 45-135 degrees
            # South: 135-225 degrees
            # West: 225-315 degrees
            if (angle >= 315.0 or angle < 45.0):
                counts[0] += 1  # North
            elif 45.0 <= angle < 135.0:
                counts[1] += 1  # East (but counted as South for symmetry)
            elif 135.0 <= angle < 225.0:
                counts[2] += 1  # South (but counted as East for symmetry)
            else:  # 225.0 <= angle < 315.0
                counts[3] += 1  # West
        
        return counts
    
    def _compute_average_speeds_per_approach(self, speeds: dict, directions: dict) -> np.ndarray:
        speed_sums = np.zeros(4, dtype=np.float32)  # [N, S, E, W]
        speed_counts = np.zeros(4, dtype=np.float32)
        
        for vehicle_id in speeds.keys():
            if vehicle_id not in directions:
                continue
                
            angle = directions[vehicle_id]
            speed = speeds[vehicle_id]
            
            if (angle >= 315.0 or angle < 45.0):
                speed_sums[0] += speed  # North
                speed_counts[0] += 1
            elif 45.0 <= angle < 135.0:
                speed_sums[1] += speed  # East
                speed_counts[1] += 1
            elif 135.0 <= angle < 225.0:
                speed_sums[2] += speed  # South
                speed_counts[2] += 1
            else:  # 225.0 <= angle < 315.0
                speed_sums[3] += speed  # West
                speed_counts[3] += 1

        avg_speeds = np.zeros(4, dtype=np.float32)
        for i in range(4):
            if speed_counts[i] > 0:
                avg_speeds[i] = speed_sums[i] / speed_counts[i]
        
        return avg_speeds
    
    def _compute_global_waiting_time(self) -> float: 
        total_waiting_time = 0.0
        try:
            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                try:
                    waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                    total_waiting_time += waiting_time
                except Exception as e:
                    logger.warning(f"Failed to get waiting time for {vehicle_id}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Failed to get vehicle ID list for waiting time: {e}")
        
        return total_waiting_time
    
    def _normalize_observation_component(self, value: float, max_value: float) -> float: 
        if max_value <= 0:
            return 0.0
        
        normalized = value / max_value
        return np.clip(normalized, 0.0, 1.0)

    def feature_reset(self):
        self._last_brake_step.clear()
        self._previous_waiting_time = 0.0
        self._vehicles_passed_count = 0
        self._last_vehicle_set = set()

    # the distance growing is there for when theh gap between cars is rather small but currently growing so there's no risk of collision

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
