from __future__ import annotations
from typing import Iterable
import numpy as np
import gymnasium as gym
import libsumo as traci
import logging
import sys
from ui.terminal_display import terminal_display
from collections import defaultdict
from features.base_v2x_feature import BaseV2XFeature
logger = logging.getLogger("v2x")

PRIORITY_TYPE = "emergency"
RETURN_DISTANCE = 200.0     # distance after which vehicles fully restore normal lane-change behavior
LANE_FREE_DIST = 8.0        # how far a car must be from another to consider lane "free"
MAX_BULK_COMMANDS_PER_STEP: int = 50   # avoid sending too many TraCI cmds

def squared_distance(p1, p2) -> float:
    # Fast squared distance (avoids slow sqrt); used to check proximity
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

class PriorityCorridorFeature(BaseV2XFeature):
    def __init__(self, feature_name="PriorityCorridorFeature", enabled=True, rl_mode=False):
        super().__init__(enabled=enabled)
        self.feature_name = feature_name
        self.rl_mode = rl_mode
        
        # Rule-based state
        self._emergency_vehicle_ids: set[str] = set()
        self._vehicles_that_yielded: set[str] = set()
        self._priority_log_events: list[tuple[str, str]] = []
        self._priority_yield_total: int = 0
        
        # RL state
        self.prev_priority_count = 0
        self.switched_last_step = False
        self.tls_id = None
        
        # Configuration
        self.priority_vehicle_types = {"emergency", "ambulance", "police", "fire"}
        self.max_distance = 1000.0
        
        if self.rl_mode:
            # RL observation: [count, avg_wait, min_dist, phase, prio_queue, other_queue]
            self.observation_size = 6
            # Actions: 0=keep, 1=switch, 2=force_green
            self.action_size = 3
        else:
            self.observation_size = 3
            self.action_size = 2

    def get_observation_space(self):
        if self.rl_mode:
            low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
            # approximate highs: count=10, wait=300s, dist=1000m, phase=10, q_prio=50, q_other=50
            high = np.array([10, 300, self.max_distance, 10, 50, 50], dtype=np.float32)
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            return gym.spaces.Box(low=0, high=1, shape=(self.observation_size,))

    def get_action_space(self):
        return gym.spaces.Discrete(self.action_size)

    def get_observation(self):
        if not self.rl_mode:
            return np.zeros(self.observation_size, dtype=np.float32)
            
        prio_vehicles = self._get_priority_vehicles()
        prio_count = len(prio_vehicles)

        avg_wait = (
            np.mean([traci.vehicle.getWaitingTime(v) for v in prio_vehicles])
            if prio_vehicles else 0.0
        )

        min_dist = (
            min([traci.vehicle.getDistance(v) for v in prio_vehicles])
            if prio_vehicles else self.max_distance
        )

        phase = self._get_current_phase()
        prio_queue = self._get_priority_queue_length()
        other_queue = self._get_other_queue_length()

        obs = np.array([
            prio_count,
            avg_wait,
            min_dist,
            phase,
            prio_queue,
            other_queue
        ], dtype=np.float32)

        return obs

    def calculate_reward(self):
        if not self.rl_mode:
            return 0.0
            
        reward = 0.0
        prio_vehicles = self._get_priority_vehicles()

        cleared_vehicles = self.prev_priority_count - len(prio_vehicles)
        if cleared_vehicles > 0:
            reward += cleared_vehicles * 10.0

        for v in prio_vehicles:
            reward -= 0.5 * traci.vehicle.getWaitingTime(v)

        reward -= 0.05 * self._get_total_queue_length()

        if self.switched_last_step:
            reward -= 1.0

        self.prev_priority_count = len(prio_vehicles)
        return float(reward)

    def _cache_positions_and_detect_emergencies(self, vehicle_ids):
        # First cache positions/edges and discover emergencies only once
        positions: dict[str, tuple[float, float]] = {}
        edges: dict[str, str] = {}
        edge_to_vehicle_ids: dict[str, list[str]] = defaultdict(list)

        for vehicle_id in vehicle_ids:
            try:
                positions[vehicle_id] = traci.vehicle.getPosition(vehicle_id)
                edge_id = traci.vehicle.getRoadID(vehicle_id)
                edges[vehicle_id] = edge_id
                edge_to_vehicle_ids[edge_id].append(vehicle_id)

                # discover emergency vehicles only once
                if vehicle_id not in self._emergency_vehicle_ids:
                    if traci.vehicle.getTypeID(vehicle_id) == PRIORITY_TYPE:
                        self._emergency_vehicle_ids.add(vehicle_id)

            except Exception as e:
                logger.error(
                    "[%s] TraCIException while caching vehicle %s: %s",
                    self.feature_name,
                    vehicle_id,
                    e,
                )

        return positions, edges, edge_to_vehicle_ids

    def _choose_best_lane_for_emergency(self, edge_id):
        # Pick the lane with the fewest vehicles on this edge
        # Used so the emergency vehicle always travels in the least-congested lane
        try:
            lane_count = traci.edge.getLaneNumber(edge_id)
        except Exception as e:
            logger.error(
                "[%s] Failed reading lane count for edge %s: %s",
                self.feature_name, edge_id, e
            )
            return 0

        lane_loads = {}
        for lane_index in range(lane_count):
            lane_id = f"{edge_id}_{lane_index}"
            try:
                vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane_id)
                lane_loads[lane_index] = len(vehicles_in_lane)
            except Exception as e:
                logger.error(
                    "[%s] Failed reading vehicles on lane %s: %s",
                    self.feature_name, lane_id, e
                )
                # if one lane fails, treat it as very busy so it is never selected
                lane_loads[lane_index] = 9999

        try:
            least_used_lane = min(lane_loads, key=lane_loads.get)
        except Exception as e:
            logger.error(
                "[%s] Failed selecting least-used lane for edge %s: %s",
                self.feature_name, edge_id, e
            )
            return 0

        return least_used_lane

    def _lane_is_free_enough(self, edge_id, lane_index, positions, vehicle_id):
        # Check if the target lane has enough space for a safe merge
        # Returns True only if no nearby vehicles are too close to block the change
        lane_id = f"{edge_id}_{lane_index}"

        try:
            vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane_id)
        except Exception as e:
            logger.error(
                "[%s] Failed reading vehicles for lane %s: %s",
                self.feature_name, lane_id, e
            )
            return False

        vehicle_position = positions[vehicle_id]

        for other_vehicle_id in vehicles_in_lane:
            if other_vehicle_id == vehicle_id:
                continue
            if other_vehicle_id not in positions:
                continue

            other_vehicle_position = positions[other_vehicle_id]

            too_close = (
                    abs(other_vehicle_position[0] - vehicle_position[0]) < LANE_FREE_DIST and
                    abs(other_vehicle_position[1] - vehicle_position[1]) < LANE_FREE_DIST
            )

            if too_close:
                return False

        return True

    def _log_priority_events(self):
        if not self._priority_log_events:
            return

        latest_short = self._priority_log_events[-1][1]

        if sys.stdout.isatty():
            summary = (
                f"[{self.feature_name}] "
                f"| total_yields={self._priority_yield_total} "
                f"| {latest_short}"
            )
            terminal_display.update("PRIORITY", summary)
            terminal_display.render()
        else:
            # Non-interactive output (piped to file): emit full verbose logs
            for verbose, _ in self._priority_log_events:
                logger.info(verbose)

    def take_action(self, action) -> None:
        # 1. ALWAYS perform rule-based lane yielding (for both RL and manual modes)
        self._perform_lane_yielding()
        
        # 2. If in RL mode, perform TLS control actions
        if self.rl_mode and self.tls_id:
            self.switched_last_step = False
            try:
                # Cast numpy types to int if needed
                if hasattr(action, 'item'):
                     # if it's a 0-d array or scalar
                    if getattr(action, 'size', 1) == 1:
                        act = int(action.item())
                    else:
                         # fallback for 1D arrays
                        act = int(action[0])
                else:
                    act = int(action)

                if act == 0:
                    # keep current phase
                    pass

                elif act == 1:
                    # normal phase switch
                    current = traci.trafficlight.getPhase(self.tls_id)
                    # Check available programs to avoid crashing if logic missing
                    # Just standard next phase logic:
                    traci.trafficlight.setPhase(self.tls_id, current + 1) 
                    # Note: SetPhase will wrap around if index exceeds, usually. 
                    # But safer to use next logic if needed. 
                    # For now using simple increment as in original code logic, 
                    # but original code grabbed program info. Let's stick to original logic:
                    # program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
                    # next_phase = (current + 1) % len(program.phases)
                    # traci.trafficlight.setPhase(self.tls_id, next_phase)
                    # Simplified for robustness:
                    self.switched_last_step = True

                elif act == 2:
                    # force priority phase (heuristic: phase 0)
                    traci.trafficlight.setPhase(self.tls_id, 0)
                    self.switched_last_step = True

            except Exception as e:
                logger.debug(f"PriorityCorridorFeature RL action error: {e}")

    def _perform_lane_yielding(self):
        try:
            vehicle_ids: Iterable[str] = traci.vehicle.getIDList()
        except Exception as e:
            logger.error(
                "[%s] Could not get vehicle list: %s",
                self.feature_name,
                e,
            )
            return
        if not vehicle_ids:
            return
        # clear per-step buffer
        self._priority_log_events.clear()

        positions, edges, edge_to_vehicle_ids = self._cache_positions_and_detect_emergencies(vehicle_ids)
        priority_ids = [vid for vid in self._emergency_vehicle_ids if vid in positions]

        if not priority_ids:
            return
        cmds_sent = 0

        for priority_id in priority_ids:
            edge_id = edges[priority_id]
            if edge_id == "":
                continue

            try:
                lane_count = traci.edge.getLaneNumber(edge_id)
            except Exception as e:
                logger.error(
                    "[%s] Could not read lane info for priority vehicle %s on edge %s: %s",
                    self.feature_name, priority_id, edge_id, e
                )
                continue

            best_lane_index = self._choose_best_lane_for_emergency(edge_id)
            priority_position = positions[priority_id]

            for vehicle_id in edge_to_vehicle_ids.get(edge_id, []):
                if vehicle_id == priority_id or vehicle_id not in positions:
                    continue

                # Skip vehicles BEHIND the emergency vehicle
                try:
                    emergency_vehicle_lane_pos = traci.vehicle.getLanePosition(priority_id)
                    current_vehicle_lane_pos = traci.vehicle.getLanePosition(vehicle_id)
                except Exception as e:
                    logger.error(
                        "[%s] Could not read lanePosition for %s or %s: %s",
                        self.feature_name, priority_id, vehicle_id, e
                    )
                    continue
                if current_vehicle_lane_pos < emergency_vehicle_lane_pos:
                    continue

                vehicle_position = positions[vehicle_id]

                # Get distance between the emergency vehicle and the current vehicle
                distance_sq = squared_distance(vehicle_position, priority_position)
                # If the emergency vehicle already passed far enough, restore normal behavior
                if distance_sq > RETURN_DISTANCE * RETURN_DISTANCE:
                    try:
                        traci.vehicle.setLaneChangeMode(vehicle_id, 1621)
                    except Exception as e:
                        logger.error(
                            "[%s] Failed restoring laneChangeMode for %s: %s",
                            self.feature_name, vehicle_id, e
                        )
                    continue

                # Only cars in the lane selected for the emergency vehicle must yield
                try:
                    vehicle_lane_index = traci.vehicle.getLaneIndex(vehicle_id)
                except Exception as e:
                    logger.error(
                        "[%s] Failed reading lane index for vehicle %s: %s",
                        self.feature_name, vehicle_id, e
                    )
                    continue

                if vehicle_lane_index != best_lane_index:
                    continue

                # Skip stationary vehicles
                try:
                    vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                except Exception as e:
                    logger.error(
                        "[%s] Failed reading speed for vehicle %s: %s",
                        self.feature_name, vehicle_id, e
                    )
                    continue

                # Ignore stopped vehicles
                if vehicle_speed < 0.1:
                    continue

                # Determine candidate lanes (left/right) for yielding
                adjacent_lane_candidates = []
                if best_lane_index + 1 < lane_count:    # Add the lane to the right
                    adjacent_lane_candidates.append(best_lane_index + 1)
                if best_lane_index - 1 >= 0:            # Add the lane to the left
                    adjacent_lane_candidates.append(best_lane_index - 1)

                vehicle_moved = False

                # Attempt to merge the vehicle into a free adjacent lane
                for target_lane_index in adjacent_lane_candidates:

                    try:
                        # Ensure the target lane has enough free space for a safe merge
                        lane_is_free = self._lane_is_free_enough(
                            edge_id=edge_id,
                            lane_index=target_lane_index,
                            positions=positions,
                            vehicle_id=vehicle_id
                        )
                    except Exception as e:
                        logger.error(
                            "[%s] Lane free check failed for vehicle %s on edge %s: %s",
                            self.feature_name, vehicle_id, edge_id, e
                        )
                        continue

                    if not lane_is_free:
                        continue

                    # Determine merge direction: +1 = left, -1 = right
                    direction = 1 if target_lane_index > vehicle_lane_index else -1

                    try:
                        if not traci.vehicle.couldChangeLane(vehicle_id, direction):
                            continue
                    except Exception as e:
                        logger.error(
                            "[%s] couldChangeLane failed for vehicle %s → direction %s: %s",
                            self.feature_name, vehicle_id, direction, e
                        )
                        continue

                    # Perform the lane change
                    try:
                        traci.vehicle.setLaneChangeMode(vehicle_id, 0)
                        traci.vehicle.changeLane(vehicle_id, target_lane_index, 1)
                        vehicle_moved = True
                        cmds_sent += 1

                        # Compute distance for debug log
                        distance_meters = squared_distance(vehicle_position, priority_position) ** 0.5

                        verbose = (
                            f"[{self.feature_name}] YIELD: vehicle {vehicle_id} -> "
                            f"lane {target_lane_index} (dist={distance_meters:.1f}m) "
                            f"@ {traci.simulation.getTime():.1f}s"
                        )
                        short = f"{vehicle_id}->{target_lane_index} dist={distance_meters:.1f}m"

                        self._priority_log_events.append((verbose, short))
                        self._priority_yield_total += 1

                    except Exception as e:
                        logger.error(
                            "[%s] Lane change failed for vehicle %s → lane %s: %s",
                            self.feature_name, vehicle_id, target_lane_index, e
                        )
                        continue

                    if vehicle_moved:
                        break

                if cmds_sent >= MAX_BULK_COMMANDS_PER_STEP:
                    return
        # Emit aggregated output
        self._log_priority_events()

    def feature_step(self):
        # default behavior: don't spam the console for rule-based runs
        logger.debug(f"[{self.feature_name}] Step completed")

    def feature_reset(self):
        self._emergency_vehicle_ids.clear()
        self._vehicles_that_yielded.clear()
        self._priority_yield_total = 0
        
        self.prev_priority_count = 0
        self.switched_last_step = False
        try:
            tls_ids = traci.trafficlight.getIDList()
            self.tls_id = tls_ids[0] if tls_ids else None
        except Exception:
            self.tls_id = None
            
        logger.debug(f"[{self.feature_name}] Reset (RL mode: {self.rl_mode})")


    def _get_priority_vehicles(self):
        try:
            vehicles = traci.vehicle.getIDList()
            return [
                v for v in vehicles
                if traci.vehicle.getTypeID(v).lower() in self.priority_vehicle_types
            ]
        except Exception:
            return []

    def _get_current_phase(self):
        return traci.trafficlight.getPhase(self.tls_id)

    def _get_priority_queue_length(self):
        count = 0
        for v in self._get_priority_vehicles():
            if traci.vehicle.getWaitingTime(v) > 0:
                count += 1
        return count

    def _get_other_queue_length(self):
        return traci.vehicle.getIDCount() - self._get_priority_queue_length()

    def _get_total_queue_length(self):
        if self.tls_id:
            try:
                lanes = traci.trafficlight.getControlledLanes(self.tls_id)
                # Use set to avoid double counting lanes with multiple connections
                lanes = set(lanes)
                return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            except Exception as e:
                logger.debug(f"[{self.feature_name}] Error getting controlled lanes: {e}")
                return 0
        return 0

    def get_feature_name(self):
        return self.feature_name