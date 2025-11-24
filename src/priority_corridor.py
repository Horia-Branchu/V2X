from __future__ import annotations
from typing import Iterable
import numpy as np
import gymnasium as gym
import libsumo as traci
import logging
from collections import defaultdict
from base_v2x_feature import BaseV2XFeature
logger = logging.getLogger("v2x.features")

TRIGGER_DISTANCE: float = 100.0          # meters
SLOWDOWN_FACTOR: float = 0.75           # 75% of current speed while yielding
YIELD_DURATION_STEPS: int = 6           # seconds to keep lane change order
# Safety limits
MIN_SPEED_AFTER_SLOWDOWN: float = 0.0   # allow full stop
MAX_BULK_COMMANDS_PER_STEP: int = 50   # avoid sending too many TraCI cmds
PRIORITY_TYPES = "emergency"

def _euclid2(p1, p2) -> float:
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
        # internal caches
        self._known_priority_ids: set[str] = set()  # discovered emergency vehicles
        self._yielded: set[str] = set()             # vehicles that already yielded
    def get_observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(self.observation_size,))

    def get_action_space(self):
        return gym.spaces.Discrete(self.action_size)

    def get_observation(self):
        dummy_obs = [0.1, 0.2, 0.3]  # dummy observation data
        logger.debug(f"[{self.feature_name}] Observation: {dummy_obs}")
        return np.array(dummy_obs)

    def calculate_reward(self):
        dummy_reward = 0.5  # dummy reward
        logger.debug(f"[{self.feature_name}] Reward: {dummy_reward}")
        return dummy_reward

    def take_action(self, action) -> None:
        if not self.enable:
            return
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
        #First cache positions/edges and discover emergencies only once
        positions: dict[str, tuple[float, float]] = {}
        edges: dict[str, str] = {}
        edge_to_vehicle_ids: dict[str, list[str]] = defaultdict(list)
        for vehicle_id in vehicle_ids:
            try:
                positions[vehicle_id] = traci.vehicle.getPosition(vehicle_id)
                edge_id = traci.vehicle.getRoadID(vehicle_id)
                edges[vehicle_id] = edge_id
                edge_to_vehicle_ids[edge_id].append(vehicle_id)

                # discover new emergency vehicles only once
                if vehicle_id not in self._known_priority_ids:
                    if traci.vehicle.getTypeID(vehicle_id) == PRIORITY_TYPES:
                        self._known_priority_ids.add(vehicle_id)
            except Exception as e:
                logger.error(
                    "[%s] TraCIException while caching vehicle %s: %s",
                    self.feature_name,
                    vehicle_id,
                    e,
                )
                continue
        priority_ids = [vid for vid in self._known_priority_ids if vid in positions]
        if not priority_ids:
            return
        cmds_sent = 0
        trigger_distance_sq = TRIGGER_DISTANCE * TRIGGER_DISTANCE

        for priority_id in priority_ids:
            if priority_id not in positions or priority_id not in edges:
                continue

            priority_edge_id = edges[priority_id]
            priority_pos = positions[priority_id]

            # lane index and lane count for this edge
            try:
                emergency_lane_index = traci.vehicle.getLaneIndex(priority_id)
                lane_count = traci.edge.getLaneNumber(priority_edge_id)
            except Exception as e:
                logger.error(
                    "[%s] TraCIException while reading lane info for emergency %s on edge %s: %s",
                    self.feature_name,
                    priority_id,
                    priority_edge_id,
                    e,
                )
                continue

            for vehicle_id in edge_to_vehicle_ids.get(priority_edge_id, ()):
                if (
                        vehicle_id == priority_id
                        or vehicle_id in self._known_priority_ids
                        or vehicle_id in self._yielded
                        or vehicle_id not in positions
                ):
                    continue

                # Distance gate
                dist_sq = _euclid2(positions[vehicle_id], priority_pos)
                if dist_sq > trigger_distance_sq:
                    continue

                try:
                    vehicle_lane_index = traci.vehicle.getLaneIndex(vehicle_id)

                    # Only move cars that are in the same lane as the ambulance
                    if vehicle_lane_index != emergency_lane_index:
                        continue

                    # If there is only one lane on this edge, there is nowhere to move
                    if lane_count <= 1:
                        continue

                    # Choose a valid target lane (away from emergency lane)
                    if emergency_lane_index > 0:
                        # emergency vehicle is not in lane 0, so move cars to the right (lower index)
                        target_lane_index = emergency_lane_index - 1
                    else:
                        # emergency vehicle is in lane 0, so move cars to lane 1
                        target_lane_index = emergency_lane_index + 1

                    traci.vehicle.changeLane(vehicle_id, target_lane_index, YIELD_DURATION_STEPS)

                    current_speed = traci.vehicle.getSpeed(vehicle_id)
                    new_speed = max(
                        MIN_SPEED_AFTER_SLOWDOWN,
                        current_speed * SLOWDOWN_FACTOR,
                    )
                    traci.vehicle.setSpeedMode(vehicle_id, 0)
                    traci.vehicle.setSpeed(vehicle_id, new_speed)

                    # remember that this vehicle already yielded
                    self._yielded.add(vehicle_id)

                    logger.info(
                        f"[{self.feature_name}] YIELD: {vehicle_id} -> lane {target_lane_index} "
                        f"(dist={dist_sq ** 0.5:.1f}m, new_speed={new_speed:.2f}) "
                        f"@ {traci.simulation.getTime():.1f}s"
                    )

                    cmds_sent += 1
                    if cmds_sent >= MAX_BULK_COMMANDS_PER_STEP:
                        return
                except Exception as e:
                    logger.error(
                        "[%s] TraCIException while yielding vehicle %s on edge %s: %s",
                        self.feature_name,
                        vehicle_id,
                        priority_edge_id,
                        e,
                    )
                    continue

    def feature_step(self):
        # default behavior: don't spam the console for rule-based runs
        logger.debug(f"[{self.feature_name}] Step completed")

    def feature_reset(self):
        logger.debug(f"[{self.feature_name}] Reset")

    def get_feature_name(self):
        return self.feature_name