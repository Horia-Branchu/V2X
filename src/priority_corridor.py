from __future__ import annotations
from typing import Iterable
import numpy as np
import gymnasium as gym
import traci
import logging
import time
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
        # performance counters (steps/time)
        self._perf_start_time = None
        self._perf_step_counter: int = 0
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
        # performance measurement(real-time speed of simulation)
        if self._perf_start_time is None:
            # start timing on first step
            self._perf_start_time = time.time()

        self._perf_step_counter += 1

        # compute real time
        elapsed = time.time() - self._perf_start_time

        # log every 50 steps to avoid spam
        if elapsed > 0 and self._perf_step_counter % 50 == 0:
            steps_per_sec = self._perf_step_counter / elapsed
            steps_per_min = steps_per_sec * 60
            logger.info(
                f"[{self.feature_name}] PERF: real_time={elapsed:.1f}s, "
                f"steps={self._perf_step_counter}, "
                f"steps/sec={steps_per_sec:.2f}, "
                f"steps/min={steps_per_min:.1f}"
            )

        veh_ids: Iterable[str] = traci.vehicle.getIDList()
        if not veh_ids:
            return
        #First cache positions/edges and discover emergencies only once
        pos: dict[str, tuple[float, float]] = {}
        edge: dict[str, str] = {}
        edge_to_veh: dict[str, list[str]] = defaultdict(list)
        for vid in veh_ids:
            try:
                pos[vid] = traci.vehicle.getPosition(vid)
                edge[vid] = traci.vehicle.getRoadID(vid)
                edge_to_veh[edge[vid]].append(vid)

                # discover new emergency vehicles only once
                if vid not in self._known_priority_ids:
                    if traci.vehicle.getTypeID(vid) == PRIORITY_TYPES:
                        self._known_priority_ids.add(vid)
            except traci.TraCIException:
                continue
        priors = [vid for vid in self._known_priority_ids if vid in pos]
        if not priors:
            return
        cmds_sent = 0
        thr2 = TRIGGER_DISTANCE * TRIGGER_DISTANCE

        for p in priors:
            if p not in pos or p not in edge:
                continue

            p_edge = edge[p]
            p_pos = pos[p]

            # lane index and lane count for this edge
            try:
                amb_lane_index = traci.vehicle.getLaneIndex(p)
                lane_count = traci.edge.getLaneNumber(p_edge)
            except traci.TraCIException:
                continue

            for v in edge_to_veh.get(p_edge, ()):
                if v == p:
                    continue
                if v in self._known_priority_ids:
                    continue
                if v in self._yielded:
                    # already yielded once, no need to spam TraCI
                    continue
                if v not in pos:
                    continue

                # Distance gate
                dist2 = _euclid2(pos[v], p_pos)
                if dist2 > thr2:
                    continue

                try:
                    veh_lane = traci.vehicle.getLaneIndex(v)

                    # Only move cars that are in the same lane as the ambulance
                    if veh_lane != amb_lane_index:
                        continue

                    # Choose a valid target lane (away from ambulance)
                    target_lane = None

                    if amb_lane_index > 0:
                        # try move right (lower index) if it exists
                        if amb_lane_index - 1 >= 0:
                            target_lane = amb_lane_index - 1
                    else:
                        # ambulance in lane 0 → try lane 1 if the edge has >1 lane
                        if amb_lane_index + 1 < lane_count:
                            target_lane = amb_lane_index + 1

                    # No valid lane to move to (single-lane edge)
                    if target_lane is None or target_lane == veh_lane:
                        continue

                    traci.vehicle.changeLane(v, target_lane, YIELD_DURATION_STEPS)

                    current_speed = traci.vehicle.getSpeed(v)
                    new_speed = max(
                        MIN_SPEED_AFTER_SLOWDOWN,
                        current_speed * SLOWDOWN_FACTOR,
                    )
                    traci.vehicle.setSpeedMode(v, 0)
                    traci.vehicle.setSpeed(v, new_speed)

                    # remember that this vehicle already yielded
                    self._yielded.add(v)

                    logger.info(
                        f"[{self.feature_name}] YIELD: {v} -> lane {target_lane} "
                        f"(dist={dist2 ** 0.5:.1f}m, new_speed={new_speed:.2f}) "
                        f"@ {traci.simulation.getTime():.1f}s"
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