import numpy as np
import gymnasium as gym
import logging
import sys
import shutil
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

        self.observation_size = 3
        self.action_size = 1
        self._last_brake_step = {}

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

    # O(n), utilizes get leader function from TRACI api, if there is no leader ahead of a car then it skips the checks for it
    # changes were made to differentiate between situations that trigger any sort of slowdown so that the logs are clearer
    def take_action(self, action):
        if not self.enable:
            return

        current_time_s = traci.simulation.getTime()

        # Collect events this step to avoid spamming logs. For interactive
        # terminals we print one concise updating line; for non-TTY we emit
        # the original verbose INFO lines so logs remain complete.
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

        # Emit aggregated output: TTY -> single updating line; non-TTY -> verbose INFOs
        any_events = any(len(v) for v in events.values())
        if any_events:
            # Update the shared terminal display so env line stays on top.
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
                # Non-interactive: still emit verbose entries for full logs
                for typ in ("EMERGENCY_BRAKE", "PREEMPTIVE_SLOWDOWN", "WARN"):
                    for verbose, _ in events[typ]:
                        logger.info(verbose)

    def feature_reset(self):
        self._last_brake_step.clear()

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
