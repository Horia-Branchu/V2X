# src/rl/priority_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time

# --- CONFIG ---
MAX_QUEUE = 50      # max vehicles counted per lane
NUM_LANES = 4
PRIORITY_VEHICLE_TYPES = {"emergency", "ambulance", "police", "bus"}

# --- Observation / Action space functions (explicit) ---
def get_observation_space(num_lanes=NUM_LANES):
    # vector: [queue_lane_0 .. queue_lane_{L-1}, prio_present_lane_0 .. prio_present_lane_{L-1},
    #          phase_id, elapsed_in_phase]
    obs_dim = num_lanes + num_lanes + 1 + 1
    low = np.zeros(obs_dim, dtype=np.float32)
    # queues up to MAX_QUEUE, prio flags [0,1], phase up to 10, elapsed up to 120s
    high = np.concatenate([
        np.full(num_lanes, MAX_QUEUE, dtype=np.float32),
        np.ones(num_lanes, dtype=np.float32),
        np.array([10.0], dtype=np.float32),
        np.array([120.0], dtype=np.float32)
    ])
    return spaces.Box(low=low, high=high, dtype=np.float32)

def get_action_space():
    # 0 = keep/extend current green
    # 1 = switch to next normal phase
    # 2 = force priority (if any priority vehicle is waiting)
    return spaces.Discrete(3)

# --- Gym-like environment wrapper for SUMO / TraCI ---
class PriorityTLSEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, traci_conn, tls_id, lane_ids, phase_list, time_step=1.0):
        """
        traci_conn: active TraCI connection object (module)
        tls_id: traffic light id controlled by this agent
        lane_ids: list of incoming lane ids (strings)
        phase_list: list of phase indices in TLS that are allowed (or mapping)
        time_step: seconds per env.step
        """
        super().__init__()
        self.traci = traci_conn
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self.phase_list = phase_list
        self.num_lanes = len(lane_ids)
        self.time_step = time_step

        self.observation_space = get_observation_space(self.num_lanes)
        self.action_space = get_action_space()

        # internal state
        self.current_phase = None
        self.elapsed_in_phase = 0.0
        self.prev_priority_wait = 0.0
        self.switch_count = 0

    # ---------- required low-level pieces (multiple functions) ----------
    def _count_queue(self):
        """Return queue length per lane using TraCI. Fallback to zeros if not available."""
        queues = []
        for lid in self.lane_ids:
            try:
                # using TraCI: number of halted vehicles or length
                q = self.traci.lane.getLastStepHaltingNumber(lid)
            except Exception:
                q = 0
            q = min(q, MAX_QUEUE)
            queues.append(float(q))
        return queues

    def _detect_priority_in_lane(self):
        """Return a binary list per lane: 1 if a priority vehicle is approaching or waiting."""
        flags = []
        for lid in self.lane_ids:
            present = 0.0
            try:
                # get vehicles on lane then check type (adapt to your vehicle type usage)
                vehs = self.traci.lane.getLastStepVehicleIDs(lid)
                for v in vehs:
                    vtype = None
                    try:
                        vtype = self.traci.vehicle.getTypeID(v)
                    except Exception:
                        pass
                    if vtype and vtype.lower() in PRIORITY_VEHICLE_TYPES:
                        present = 1.0
                        break
            except Exception:
                present = 0.0
            flags.append(present)
        return flags

    def _collect_observation(self):
        queues = self._count_queue()
        prio_flags = self._detect_priority_in_lane()
        # read TLS state
        try:
            self.current_phase = int(self.traci.trafficlight.getPhase(self.tls_id))
        except Exception:
            self.current_phase = 0
        # elapsed time - if you track it elsewhere, adapt
        obs = np.array(queues + prio_flags + [float(self.current_phase), float(self.elapsed_in_phase)], dtype=np.float32)
        return obs

    def _apply_action(self, action):
        """
        Apply the chosen action using TraCI.
        - action 0: keep current (extend)
        - action 1: switch to next phase
        - action 2: force a priority-friendly phase (if defined)
        """
        # this is where you must align with your project's TLS control API
        if action == 0:
            # extend - do nothing special: just allow the TLS to keep green; we will advance simulation externally
            pass
        elif action == 1:
            # next phase: increment phase index
            try:
                next_phase = (self.current_phase + 1) % len(self.phase_list)
                self.traci.trafficlight.setPhase(self.tls_id, self.phase_list[next_phase])
                self.switch_count += 1
                self.elapsed_in_phase = 0.0
            except Exception:
                pass
        elif action == 2:
            # forcing a priority phase: you must have a mapping from priority lanes -> phase index
            # example: if phase 2 gives green to lane0 (priority), set it
            # We'll attempt to find a phase that serves a lane with a waiting priority vehicle
            prio_flags = self._detect_priority_in_lane()
            try:
                # naive mapping: phase_list indices map to lanes 0..n-1 in order (adapt as needed)
                for i, f in enumerate(prio_flags):
                    if f > 0.5 and i < len(self.phase_list):
                        self.traci.trafficlight.setPhase(self.tls_id, self.phase_list[i])
                        self.switch_count += 1
                        self.elapsed_in_phase = 0.0
                        break
            except Exception:
                pass
        else:
            pass

    def _compute_reward(self, prev_state, action, new_state):
        """
        Reward design:
        - big positive (+R) when a priority vehicle passes (we detect via decrease in waiting/vehicles)
        - penalty for priority waiting time
        - penalty for total waiting queue
        - penalty for too many switches
        """
        # extract useful parts
        prev_queues = prev_state[:self.num_lanes]
        prev_prio = prev_state[self.num_lanes:self.num_lanes*2]
        new_queues = new_state[:self.num_lanes]
        new_prio = new_state[self.num_lanes:self.num_lanes*2]

        # Track priority vehicles that left: if a lane had prio before and now queue decreased -> reward
        prio_passed = 0
        prio_wait_penalty = 0.0
        for i in range(self.num_lanes):
            if prev_prio[i] > 0.5:
                # if number of vehicles in that lane decreased => some passed
                if new_queues[i] < prev_queues[i]:
                    prio_passed += (prev_queues[i] - new_queues[i])
                else:
                    # still waiting: penalize
                    prio_wait_penalty += 1.0

        total_wait = float(sum(new_queues))
        # reward terms: tune weights to your scenario
        reward = 0.0
        reward += 8.0 * float(prio_passed)         # big reward for letting priority pass
        reward -= 5.0 * prio_wait_penalty          # penalty for priority waiting
        reward -= 0.05 * total_wait                # small penalty for overall congestion
        # penalty for switching
        if action == 1:
            reward -= 0.2
        if action == 2:
            # encourage using force but don't abuse it
            reward -= 0.1

        return reward

    # ---------- Gym API ----------
    def step(self, action):
        # 1) collect previous observation (for reward shaping)
        prev_obs = self._collect_observation()

        # 2) apply action (which changes the traffic light)
        self._apply_action(action)

        # 3) advance simulation by a few TraCI steps (time_step / sim_step)
        # The simulation stepping must be done by your main loop; here we assume you provide a helper or call traci.simulationStep()
        try:
            steps = int(max(1, self.time_step))
            for _ in range(steps):
                self.traci.simulationStep()
        except Exception:
            # fallback: sleep to simulate time progression while not connected
            time.sleep(0.001)

        # 4) update elapsed time
        self.elapsed_in_phase += self.time_step

        # 5) collect new observation
        new_obs = self._collect_observation()

        # 6) compute reward
        reward = self._compute_reward(prev_obs, action, new_obs)

        # 7) done flag: episodic control is external (stop when sim finished)
        done = False

        info = {"tls_id": self.tls_id, "switch_count": self.switch_count}
        return new_obs, reward, done, False, info  # gymnasium returns (obs, reward, terminated, truncated, info)

    def reset(self, *, seed=None, options=None):
        # Optionally reset the SUMO simulation externally; here we reset env internal counters
        self.elapsed_in_phase = 0.0
        self.switch_count = 0
        obs = self._collect_observation()
        return obs, {}
