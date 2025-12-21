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
    def __init__(self, feature_name="PriorityCorridorFeature", enabled=True):
        super().__init__(enabled=enabled)
        self.feature_name = feature_name
        self.observation_size = 3  # dummy observation size
        self.action_size = 2
        self._emergency_vehicle_ids: set[str] = set()
        self._vehicles_that_yielded: set[str] = set()
        # Per-step event buffer for compact TTY display or verbose non-TTY logs
        self._priority_log_events: list[tuple[str, str]] = []
        # Cumulative number of successful yield maneuvers in this run
        self._priority_yield_total: int = 0
    def get_observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(self.observation_size,))

    def get_action_space(self):
        return gym.spaces.Discrete(self.action_size)

    def get_observation(self):
        return np.zeros(self.observation_size, dtype=np.float32)

    def calculate_reward(self):
        return 0.0

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
        logger.debug(f"[{self.feature_name}] Reset")

    def get_feature_name(self):
        return self.feature_name