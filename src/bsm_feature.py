
import numpy as np
import gymnasium as gym
import logging
import traci
from base_v2x_feature import BaseV2XFeature

logger = logging.getLogger("v2x.features")


class BSMFeature(BaseV2XFeature):
    def __init__(
        self,
        feature_name: str = "BSMFeature",
        enabled: bool = True,
        min_gap: float = 10.0,
        ttc_thresh: float = 1.5,
        leader_decel_thresh: float = -4.0,
        brake_duration: float = 1.0,
        max_react_gap: float = 60.0,
        max_ttc_gap: float = 80.0,
        max_decel_gap: float = 60.0,
    ):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.min_gap = float(min_gap)
        self.ttc_thresh = float(ttc_thresh)
        self.leader_decel_thresh = float(leader_decel_thresh)
        self.brake_duration = float(brake_duration)
        self.max_react_gap = float(max_react_gap)
        self.max_ttc_gap = float(max_ttc_gap)
        self.max_decel_gap = float(max_decel_gap)

        self.observation_size = 3
        self.action_size = 1
        self._last_brake_step = {}

    # ---- required interface ----
    def get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_size,))

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(self.action_size)

    def get_observation(self) -> np.ndarray:
        return np.zeros(self.observation_size, dtype=np.float32)

    def calculate_reward(self) -> float:
        return 0.0

    def get_feature_name(self) -> str:
        return self.feature_name

    def take_action(self, action):
        return

    # ---- main loop ----
    def feature_step(self):
        if not self.enable:
            return

        sim_t = traci.simulation.getTime()
        for vid in traci.vehicle.getIDList():
            leader = traci.vehicle.getLeader(vid, dist=self.max_react_gap)
            if not leader:
                continue
            lead_id, gap = leader
            if lead_id is None or gap < 0 or gap > self.max_react_gap:
                continue

            try:
                v_follow = traci.vehicle.getSpeed(vid)
                v_lead = traci.vehicle.getSpeed(lead_id)
                a_lead = traci.vehicle.getAcceleration(lead_id)
            except traci.TraCIException:
                continue

            rel_speed = max(0.0, v_follow - v_lead)
            ttc = (gap / rel_speed) if rel_speed > 1e-3 else float("inf")

            should_brake = (
                gap < self.min_gap
                or (gap <= self.max_ttc_gap and ttc < self.ttc_thresh)
                or (gap <= self.max_decel_gap and a_lead <= self.leader_decel_thresh)
            )

            last = self._last_brake_step.get(vid, -10.0)
            if should_brake and (sim_t - last) >= 0.5:
                self._trigger_emergency_brake(vid, gap, ttc, lead_id, sim_t)
                self._last_brake_step[vid] = sim_t

    def feature_reset(self):
        self._last_brake_step.clear()

    # ---- helpers ----
    def _trigger_emergency_brake(self, veh_id: str, gap: float, ttc: float, leader_id: str, sim_t: float):
        ttc_str = f"{ttc:.2f}s" if np.isfinite(ttc) else "distance shortened"
        logger.info(
            f"[{self.feature_name}] BSM: {veh_id} EMERGENCY_BRAKE "
            f"(leader={leader_id}, gap={gap:.1f}m, ttc={ttc_str}) @ {sim_t:.1f}s"
        )
        try:
            traci.vehicle.slowDown(veh_id, 0.0, self.brake_duration)
        except traci.TraCIException as e:
            logger.warning(f"[{self.feature_name}] brake failed for {veh_id}: {e}")
