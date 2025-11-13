from __future__ import annotations
from typing import Iterable
import numpy as np
import gymnasium as gym
import traci
import logging
from base_v2x_feature import BaseV2XFeature
logger = logging.getLogger("v2x.features")


TRIGGER_DISTANCE: float = 100.0          # meters
SLOWDOWN_FACTOR: float = 0.75           # 75% of current speed while yielding
YIELD_DURATION_STEPS: int = 6           # seconds to keep lane change order

# Safety limits
MIN_SPEED_AFTER_SLOWDOWN: float = 0.0   # allow full stop
MAX_BULK_COMMANDS_PER_STEP: int = 50   # avoid sending too many TraCI cmds


PRIORITY_TYPES = "emergency"

def _is_priority(veh_id: str) -> bool:
    try:
        return traci.vehicle.getTypeID(veh_id) == PRIORITY_TYPES
    except traci.TraCIException:
        return False


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

        veh_ids: Iterable[str] = traci.vehicle.getIDList()
        priors = [vid for vid in veh_ids if _is_priority(vid)]
        if not priors:
            return

        # Cache positions and edges to avoid repeated calls
        pos, edge = {}, {}
        for vid in veh_ids:
            try:
                pos[vid] = traci.vehicle.getPosition(vid)
                edge[vid] = traci.vehicle.getRoadID(vid)
            except traci.TraCIException:
                continue

        # Rightmost - index 0 in SUMO
        cmds_sent = 0
        for p in priors:
            if p not in pos or p not in edge:
                continue


            p_edge = edge[p]
            p_pos = pos[p]

            # Consider only cars on the same edge as the priority car
            for v in veh_ids:
                if v == p or _is_priority(v):
                    continue
                if v not in pos or v not in edge:
                    continue
                if edge[v] != p_edge:
                    continue

                # Distance gate
                if _euclid2(pos[v], p_pos) > (TRIGGER_DISTANCE * TRIGGER_DISTANCE):
                    continue

                # Issue yield directives: change to lane 0 and slow down
                try:
                    traci.vehicle.changeLane(v, 0, YIELD_DURATION_STEPS)
                    current_speed = traci.vehicle.getSpeed(v)
                    new_speed = max(MIN_SPEED_AFTER_SLOWDOWN, current_speed * SLOWDOWN_FACTOR)
                    traci.vehicle.setSpeedMode(v, 0)
                    traci.vehicle.setSpeed(v, new_speed)

                    logger.info(
                        f"[{self.feature_name}] YIELD: {v} -> lane 0 "
                        f"(dist={(_euclid2(pos[v], p_pos)) ** 0.5:.1f}m, "
                        f"new_speed={new_speed:.2f}) @ {traci.simulation.getTime():.1f}s"
                    )

                    cmds_sent += 1
                    if cmds_sent >= MAX_BULK_COMMANDS_PER_STEP:
                        return
                except traci.TraCIException:
                    continue

    def feature_step(self):
        # default behavior: don't spam the console for rule-based runs
        logger.debug(f"[{self.feature_name}] Step completed")

    def feature_reset(self):
        logger.debug(f"[{self.feature_name}] Reset")

    def get_feature_name(self):
        return self.feature_name