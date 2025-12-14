import gymnasium as gym
import libsumo as traci
import numpy as np
import logging

logger = logging.getLogger("v2x")


class PriorityEnvironment:
    """
    RL Feature for handling priority / emergency vehicles.

    """

    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight

        # internal state
        self.prev_priority_count = 0
        self.switched_last_step = False

        # configuration (adapt if needed)
        self.priority_vehicle_types = {"emergency", "ambulance", "police", "fire"}
        self.max_distance = 1000.0

        # traffic light to control (can be generalized later)
        self.tls_id = None


    # Required Feature API


    def get_feature_name(self):
        return self.name

    def feature_reset(self):
        """Reset feature state at the beginning of each episode"""
        self.prev_priority_count = 0
        self.switched_last_step = False

        try:
            tls_ids = traci.trafficlight.getIDList()
            self.tls_id = tls_ids[0] if tls_ids else None
        except Exception:
            self.tls_id = None

    def feature_step(self):
        """Executed every simulation step """
        pass


    # RL SPACES


    def get_observation_space(self):
        """
        Observation vector:
        [0] number of priority vehicles
        [1] average waiting time of priority vehicles
        [2] distance of closest priority vehicle
        [3] current traffic light phase
        [4] queue length on priority lanes
        [5] queue length on non-priority lanes
        """
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = npnd = np.array([10, 300, self.max_distance, 10, 50, 50], dtype=np.float32)

        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def get_action_space(self):
        """
        Discrete actions:
        0 - keep current phase
        1 - switch to next phase
        2 - force priority green phase
        """
        return gym.spaces.Discrete(3)


    # OBSERVATION


    def get_observation(self):
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


    # ACTION APPLICATION


    def take_action(self, action):
        if self.tls_id is None:
            return

        self.switched_last_step = False

        try:
            if action == 0:
                # keep current phase
                return

            elif action == 1:
                # normal phase switch
                current = traci.trafficlight.getPhase(self.tls_id)
                program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
                next_phase = (current + 1) % len(program.phases)
                traci.trafficlight.setPhase(self.tls_id, next_phase)
                self.switched_last_step = True

            elif action == 2:
                # force priority phase (heuristic: phase 0)
                traci.trafficlight.setPhase(self.tls_id, 0)
                self.switched_last_step = True

        except Exception as e:
            logger.debug(f"PriorityEnvironment action error: {e}")


    # REWARD FUNCTION


    def calculate_reward(self):
        """
        Reward strategy:
        +10  for each priority vehicle that passes
        -0.5 per second of waiting for priority vehicles
        -0.05 per queued vehicle (global congestion)
        -1   for unnecessary switching
        """

        reward = 0.0
        prio_vehicles = self._get_priority_vehicles()

        # reward passing priority vehicles
        passed = self.prev_priority_count - len(prio_vehicles)
        reward += passed * 10.0

        # penalize waiting time
        for v in prio_vehicles:
            reward -= 0.5 * traci.vehicle.getWaitingTime(v)

        # congestion penalty
        reward -= 0.05 * self._get_total_queue_length()

        # switching penalty
        if self.switched_last_step:
            reward -= 1.0

        self.prev_priority_count = len(prio_vehicles)

        return reward


    # HELPER FUNCTIONS (algorithm breakdown)


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
        try:
            return traci.trafficlight.getPhase(self.tls_id)
        except Exception:
            return 0

    def _get_priority_queue_length(self):
        count = 0
        try:
            for v in self._get_priority_vehicles():
                if traci.vehicle.getWaitingTime(v) > 0:
                    count += 1
        except Exception:
            pass
        return count

    def _get_other_queue_length(self):
        try:
            return traci.vehicle.getIDCount() - self._get_priority_queue_length()
        except Exception:
            return 0

    def _get_total_queue_length(self):
        try:
            lanes = traci.lane.getIDList()
            return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        except Exception:
            return 0
