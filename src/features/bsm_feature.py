import numpy as np
import gymnasium as gym
import logging
import sys
import libsumo as traci
from terminal_display import terminal_display
from base_v2x_feature import BaseV2XFeature

logger = logging.getLogger("v2x.features")

DEFAULT_LAST_BRAKE_TIME = -10.0
MIN_BRAKE_INTERVAL_S = 0.5
PREEMPTIVE_SLOWDOWN_FACTOR = 0.7

MIN_DETECTION_RANGE = 20.0
MAX_DETECTION_RANGE = 100.0
MIN_TTC_THRESHOLD = 0.5
MAX_TTC_THRESHOLD = 3.0

DEFAULT_DETECTION_RANGE = 50.0
DEFAULT_TTC_THRESHOLD = 1.5
DEFAULT_BRAKE_DURATION = 1.0

REWARD_W_TTC = 2.0
REWARD_W_BRAKE = 0.5
REWARD_W_CRITICAL = 5.0
REWARD_W_WAIT = 1.0
REWARD_WAIT_TIME_NORMALIZATION = 100.0

NUM_APPROACHES = 4
METRICS_PER_APPROACH = 3
OBSERVATION_SIZE = NUM_APPROACHES * METRICS_PER_APPROACH

MIN_GAP = 10.0
MAX_TTC_GAP = 80.0
EPSILON_SPEED = 1e-3

ACTION_DECREASE_TTC = 0
ACTION_INCREASE_TTC = 1
ACTION_DECREASE_RANGE = 2
ACTION_INCREASE_RANGE = 3
ACTION_NO_OP = 4

TTC_STEP = 0.1
RANGE_STEP = 5.0

class BSMFeature(BaseV2XFeature):

    def __init__(self, feature_name="BSMFeature", enabled=True, rl_mode=False):
        super().__init__(enabled)
        self.feature_name = feature_name
        self.rl_mode = rl_mode
        self.observation_size = OBSERVATION_SIZE
        
        self.detection_range = DEFAULT_DETECTION_RANGE
        self.ttc_threshold = DEFAULT_TTC_THRESHOLD
        self.brake_duration = DEFAULT_BRAKE_DURATION
        
        self.min_detection_range = MIN_DETECTION_RANGE
        self.max_detection_range = MAX_DETECTION_RANGE
        self.min_ttc_threshold = MIN_TTC_THRESHOLD
        self.max_ttc_threshold = MAX_TTC_THRESHOLD
        
        self.w_ttc = REWARD_W_TTC
        self.w_brake = REWARD_W_BRAKE
        self.w_critical = REWARD_W_CRITICAL
        self.w_wait = REWARD_W_WAIT
        
        self._bsm_log_events = []
        
        self._last_brake_step = {}
        self._emergency_brake_count = 0
        self._slowdown_count = 0
        
        self._subscribed_vehicles = set()

    def get_observation_space(self):
        return gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.observation_size,),
            dtype=np.float32
        )

    def get_action_space(self):
        return gym.spaces.Discrete(5)

    def get_observation(self):
        if not self.enable:
            return np.zeros(self.observation_size, dtype=np.float32)
        
        vehicle_ids = traci.vehicle.getIDList()
        
        approach_stats = np.zeros((4, 3), dtype=np.float32)
        
        for vid in vehicle_ids:
            res = traci.vehicle.getSubscriptionResults(vid)
            if not res:
                continue

            angle = res.get(traci.constants.VAR_ANGLE, 0.0)
            speed = res.get(traci.constants.VAR_SPEED, 0.0)
            wait = res.get(traci.constants.VAR_ACCUMULATED_WAITING_TIME, 0.0)
            
            if angle < 45 or angle >= 315:
                bucket = 0
            elif 45 <= angle < 135:
                bucket = 1
            elif 135 <= angle < 225:
                bucket = 2
            else:
                bucket = 3
            
            approach_stats[bucket][0] += 1.0
            approach_stats[bucket][1] += speed
            approach_stats[bucket][2] += wait

        obs = []
        for i in range(4):
            count = approach_stats[i][0]
            if count > 0:
                avg_speed = approach_stats[i][1] / count
                avg_wait = approach_stats[i][2] / count
            else:
                avg_speed = 0.0
                avg_wait = 0.0
            
            obs.extend([count, avg_speed, avg_wait])
            
        return np.array(obs, dtype=np.float32)

    def calculate_reward(self):
        if not self.enable:
            return 0.0
        
        vehicle_ids = traci.vehicle.getIDList()
        
        total_waiting_time = 0.0
        total_ttc_penalty = 0.0
        critical_count = 0
        
        for vid in vehicle_ids:
            res = traci.vehicle.getSubscriptionResults(vid)
            if not res:
                continue
                
            total_waiting_time += res.get(traci.constants.VAR_ACCUMULATED_WAITING_TIME, 0.0)

            leader_data = res.get(traci.constants.VAR_LEADER, None)
            
            if not leader_data:
                continue
                
            leader_id, gap = leader_data
            if leader_id is None or gap < 0:
                continue
            
            follower_speed = res.get(traci.constants.VAR_SPEED, 0.0)
            
            leader_res = traci.vehicle.getSubscriptionResults(leader_id)
            if leader_res:
                leader_speed = leader_res.get(traci.constants.VAR_SPEED, 0.0)
            else:
                leader_speed = traci.vehicle.getSpeed(leader_id)
            
            time_to_collision = self._calculate_time_to_collision(gap, follower_speed, leader_speed)
            
            if time_to_collision < self.ttc_threshold:
                total_ttc_penalty += (self.ttc_threshold - time_to_collision)
                critical_count += 1
        
        reward = (
            - self.w_wait * total_waiting_time / REWARD_WAIT_TIME_NORMALIZATION
            - self.w_ttc * total_ttc_penalty
            - self.w_brake * self._emergency_brake_count
            - self.w_critical * critical_count
        )
        
        return float(reward)

    def get_feature_name(self):
        return self.feature_name

    def _calculate_time_to_collision(self, gap: float, follower_speed: float, leader_speed: float) -> float:
        relative_speed_mps = max(0.0, follower_speed - leader_speed)
        
        if relative_speed_mps > EPSILON_SPEED:
            return gap / relative_speed_mps
        else:
            return float('inf')

    def bsm_message_log(self, message, event_type: str = "GENERIC"):
        timestamp = traci.simulation.getTime()
        verbose = f"[{self.feature_name}] {event_type}: {message} @ {timestamp:.1f}s"
        short = message.split(',')[0]
        self._bsm_log_events.append((verbose, short))

    def _log_bsm_events(self):
        if sys.stdout.isatty():
            if not self._bsm_log_events:
                terminal_display.update("BSM", f"[{self.feature_name}] Status: Monitoring...")
                terminal_display.render()
                return

            event_count = len(self._bsm_log_events)
            latest_short = self._bsm_log_events[-1][1]
            summary = f"[{self.feature_name}] | events={event_count} | {latest_short}"
            terminal_display.update("BSM", summary)
            terminal_display.render()
        else:
            if not self._bsm_log_events:
                return
            for verbose, _ in self._bsm_log_events:
                logger.info(verbose)

    def take_action(self, action):
        if not self.enable:
            return

        self._bsm_log_events.clear()
        self._emergency_brake_count = 0
        self._slowdown_count = 0

        self._update_subscriptions()

        if self.rl_mode:
            if action == ACTION_DECREASE_TTC:
                self.ttc_threshold = max(self.min_ttc_threshold, self.ttc_threshold - TTC_STEP)
            elif action == ACTION_INCREASE_TTC:
                self.ttc_threshold = min(self.max_ttc_threshold, self.ttc_threshold + TTC_STEP)
            elif action == ACTION_DECREASE_RANGE:
                self.detection_range = max(self.min_detection_range, self.detection_range - RANGE_STEP)
            elif action == ACTION_INCREASE_RANGE:
                self.detection_range = min(self.max_detection_range, self.detection_range + RANGE_STEP)
        
        vehicle_list = traci.vehicle.getIDList()
            
        for vid in vehicle_list:
            self.apply_bsm_safety(vid)

        self._log_bsm_events()

    def _update_subscriptions(self):
        current_ids = set(traci.vehicle.getIDList())
        new_ids = current_ids - self._subscribed_vehicles
        
        for vid in new_ids:
            traci.vehicle.subscribe(vid, [
                traci.constants.VAR_SPEED, 
                traci.constants.VAR_ANGLE, 
                traci.constants.VAR_ACCUMULATED_WAITING_TIME,
                traci.constants.VAR_LEADER
            ])
            self._subscribed_vehicles.add(vid)
            
        self._subscribed_vehicles.intersection_update(current_ids)

    def apply_bsm_safety(self, vehicle_id):
        res = traci.vehicle.getSubscriptionResults(vehicle_id)
        if not res:
            return

        leader_data = res.get(traci.constants.VAR_LEADER, None)
        if not leader_data:
            return
            
        leader_id, distance_to_leader_m = leader_data
        if leader_id is None or distance_to_leader_m < 0 or distance_to_leader_m > self.detection_range:
            return

        follower_speed_mps = res.get(traci.constants.VAR_SPEED, 0.0)
        
        leader_res = traci.vehicle.getSubscriptionResults(leader_id)
        if leader_res:
            leader_speed_mps = leader_res.get(traci.constants.VAR_SPEED, 0.0)
        else:
            leader_speed_mps = traci.vehicle.getSpeed(leader_id)

        relative_speed_mps = max(0.0, follower_speed_mps - leader_speed_mps)
        time_to_collision_s = (
            distance_to_leader_m / relative_speed_mps if relative_speed_mps > EPSILON_SPEED else float("inf")
        )

        gap_trigger = (distance_to_leader_m < MIN_GAP)
        time_to_collision_trigger = (
            distance_to_leader_m <= MAX_TTC_GAP
            and time_to_collision_s < self.ttc_threshold
        )
        
        should_brake = gap_trigger or time_to_collision_trigger

        current_time_s = traci.simulation.getTime()
        last_brake_time = self._last_brake_step.get(vehicle_id, DEFAULT_LAST_BRAKE_TIME)
        
        if should_brake and (current_time_s - last_brake_time) >= MIN_BRAKE_INTERVAL_S:
            self._emergency_brake_count += 1
            self.bsm_message_log(f"Brake triggered for {vehicle_id}", "SAFETY")
            try:
                traci.vehicle.slowDown(vehicle_id, 0.0, self.brake_duration)
                self._last_brake_step[vehicle_id] = current_time_s
            except traci.TraCIException:
                pass
            else:
                self._slowdown_count += 1

    def feature_step(self):
        self._update_subscriptions()
        logger.debug(f"[{self.feature_name}] Step")

    def feature_reset(self):
        self._bsm_log_events.clear()
        self._last_brake_step.clear()
        self._emergency_brake_count = 0
        self._slowdown_count = 0
        self._subscribed_vehicles.clear()
        logger.debug(f"[{self.feature_name}] Reset")