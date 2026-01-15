# Documentation

## Project Structure

V2X/  
├── config/ <br>
│   ├── simulation.sumocfg <br>
│   ├── generate_all_vehicles_scripts.py <br>
│   ├── time_to_run.txt <br>
├── src/ <br>
│   ├── analysis/ - Data analysis scripts  
│   │   ├── correlation_map.py  
│   │   ├── geo_emissions_plot.py  
│   │   ├── geo_plots.py  
│   │   └── plots.py  
│   ├── data/  
│   │   └── vehicles.csv  
│   ├── datacollector/  
│   │   └── data_collector.py  
│   ├── environment/  
│   │   └── [base_sumo_env.py](#base-sumo-environment)  
│   ├── features/  
│   │   ├── [base_v2x_feature.py](#base-v2x-feature)  
│   │   ├── bsm_feature.py  
│   │   ├── [dynamic_tls.py](#dynamic-tls)  
│   │   └── [priority_corridor.py](#priority-corridor-feature)  
│   ├── runners/  
│   │   ├── collector_runner.py  
│   │   ├── rl_collector_runner.py  
│   │   ├── [rl_tester.py](#rl_tester)  
│   │   ├── [rl_trainee.py](#rl_trainee)  
│   │   └── [simulation_runner.py](#simulation-runner-class)  
│   └── ui/ <br>
│       ├── [progress_bar.py](#progress-bar)  
│       └── [terminal_display.py](#terminal-display)  
├── [main.py](#main) <br>
├── [DOCUMENTATION.md](#Documentation)  



# Modules

# Base Sumo Environment
### Constructor
Initializes the traffic simulation environment.

**Input:**
- `sumo_config` (str): Path to SUMO configuration file
- `simulation_steps` (int): Total simulation steps (default: 1000)
- `gui` (bool): Enable GUI display (default: True)
- `bsm` (bool): Enable Basic Service Messages (default: False)
- `tls` (bool): Enable Traffic Light System (default: False)
- `priority` (bool): Enable priority mode (default: False)
- `reroute` (bool): Enable rerouting (default: False)
- `rl` (bool): Enable Reinforcement Learning (default: False)

**Output:** `None` (initializes object state)

**What it does:** Sets up simulation parameters, sets log level, initializes step counter, and builds SUMO command.

### _setup_spaces

**Input:**

**Output:**

**What it does:** Initializes the observation space and action space for the reinforcement learning environment. Currently sets up dummy spaces to be implemented in future RL development.

### _setup_features

**Input:**
- `bsm` (bool): Enable Basic Service Messages
- `tls` (bool): Enable Traffic Light System
- `priority` (bool): Enable priority mode
- `reroute` (bool): Enable rerouting

**Output:** `None`

**What it does:** Initializes features based on the provided flags. Creates and stores feature objects that will be used during simulation, such as BSM, TLS, priority, and reroute features.

### _build_sumo_command

**Input:** `None`

**Output:** Returns the string to call sumo in the command line

**What it does:** It initializez the sumo base command based on the arguments (can be either with the GUI or in the command line).

### reset
**Input:**
- `seed`: (int, optional): Random seed for reproducibility. (default: `None`)
- `options`:  (dict, optional): Additional reset options. (default `None`)

**Output:**
- `observation`: Initial environment observation after reset
- `info`: Dictionary containing auxiliary information

**What it does:** Closes any existing simulation, starts a new simulation instance, resets step counter, checks features, resets scenario, and returns initial observation data.

### _startup_spinner

**Input:**
- `stop_event` (threading.Event): Event to signal when to stop the spinner

**Output:** `None`

**What it does:** Displays a simple CLI spinner animation while SUMO is starting up to provide visual feedback that the simulation is loading.

### step
**Input:**
- `action`: (): Individual action the algorithm should take at any given step.

**Output:**
- `observation`:
- `reward`:
- `terminated`:
- `truncated`:
- `info`:

**What it does:** Takes the custom action, moves on with the simulation, increases the counter then it collects the results (observation, reward, terminated, truncated, info) and then returns them.

### _take_action

**Input:**
- `action`: Action to be distributed to features

**Output:** `None`

**What it does:** Distributes the given action to all active features in the environment.

### _get_observation

**Input:** `None`

**Output:** numpy array containing combined observations from all features

**What it does:** Collects and combines observation data from all active features into a single observation array.

### _calculate_reward

**Input:** `None`

**Output:** float representing the total calculated reward

**What it does:** Computes the total reward by combining weighted rewards from all active features.

### _scenario_reset

**Input:** `None`

**Output:** `None`

**What it does:** Placeholder method for scenario-specific reset logic. Can be overridden by subclasses for custom scenario initialization.

### _is_terminated

**Input:**

**Output:** `bool` indicating whether the episode has terminated

**What it does:** Placeholder method for scenario-specific termination logic. Can be overridden by subclasses to define custom termination conditions.

### _get_info

**Input:** (None)

**Output:** Dictionary only with `step` and `active_features`.

**What it does:** It builds a simple dictionary (for now) with the current step and the features enabled.

### close
**Input:** `None`

**Output:** `None`

**What it does:**: It tries to close the Traci Module and if there is any GUI opened, it terminates the process based on the current OS running.

# Base V2X Feature

### Constructor
Initializes the base V2X feature

**Input:**
- `enabled` (bool): check if the feature is enabled or not (default:True)

**Output:** `None` (initializes object state)

**What it does:** Sets the availabilty of the feature and sets the default weight

### get_observation_space
**Input:** `None`

**Output:** (gym.Space)

**What it does:** To be implemented in the child class

### get_action_space
**Input:** `None`

**Output:** (gym.Space)

**What it does:** To be implemented in the child class

### take_action
**Input:** `None`

**Output:** (None)

**What it does:** To be implemented in the child class

### get_observation
**Input:** `None`

**Output:** (np.ndarray)

**What it does:** To be implemented in the child class

### calculate_reward
**Input:** `None`

**Output:** (None)

**What it does:** To be implemented in the child class

### feature_step
**Input:** `None`

**Output:** (None)

**What it does:** To be implemented in the child class

### feature_reset
**Input:** `None`

**Output:** (None)

**What it does:** To be implemented in the child class

# Dynamic TLS

### Constructor
Initializes the DynamicTLS feature module responsible for adaptive traffic light behavior.

**Input:**
- `feature_name` (str): Name of the feature (default: "DynamicTLS").
- `enabled` (bool): Whether the feature is active (default: True).
- `rl_mode` (bool): Whether RL mode is active (default: False).

**Output:**
`None` (initializes internal state)

**What it does:** Creates and configures the dynamic traffic-light-control feature, defining observation/action sizes, vehicle detection range, green-light extension duration, parameters used for calculating the reward and bookkeeping structures for temporary TLS overrides.

### get_observation_space
Defines the observation space for the dynamic TLS feature.

**Input:** `None`

**Output:**
- `gym.spaces.Box`: An observation space represeting the state of all traffic lights.

**What it does:**
- Retrieves all traffic light IDs from the SUMO simulation.
- DeterDetermines the total number of traffic lights.
- Builds a flattened observation vector containing state information for each traffic light
- Sets lower bounds to 0.0 for all observation values.
- Stes upper bounds based on:
   - Queue length limits (20) for incoming lanes
   - A fixed maximum value (60.0) for a time-based feature
- Returns a Gym Box space suitable for reinforcement learning agents
- Observation structure per traffic light:
   - 4 values: Queue lengths for controlled lanes (clamped)
   - 1 value: Time since last green phase

### get_action_space
Defines the action space for the dynamic TLS feature.

**Input:** `None`

**Output:**
- `gym.spaces.Discrete`: An action space with values in the range [0, 1].

**What it does:**
- Creates a continuous action space represented by a 3-dimensional vector
- Each action component is bounded between 0 and 1
- Uses float32 precision for compatibility with Gym and RL algorithms
- Action structure:
   - A vector of length 3
   - Each elemnt represents a normalized control signal for the traffic light logic (maintain, extend green, switch to next phase)


### get_approaching_vehicles_by_lane(tls_id)
Groups approaching vehicles per lane within the configured detection range.

**Input:**
- `tls_id` (str): Traffic light identifier.

**Output:**
- `approaching` (Dictionary): { lane_id: [vehicle_ids...] }

**What it does:** For each controlled lane:
- Retrieves active vehicles
- Computes distance to the intersection,
- Includes only vehicles within the detection range (default: 50 m).

### get_lanes_on_same_street(tls_id, lane_id)
Finds all lanes associated with the same road segment.

**Input:**
- `tls_id` (str): Traffic light identifier.
- `lane_id` (str): Reference lane.

**Output:**

- `same_street_lanes` (List): lane IDs belonging to the same street.

**What it does:** Matches edges between the reference lane and all lanes controlled by the TLS, returning those on the same road.

### set_tls_green_for_vehicle(tls_id, v_id)
Overrides TLS signals so a specific vehicle's lane receives green.

**Input:**
- `tls_id` (str): Traffic light identifier.
- `v_id` (str): Vehicle ID.

**Output:**
- `None`

**What it does:**
- Identifies all lanes on the vehicle's street,
- Reconstructs the TLS phase string,
- Switches only those lanes to green,
- Applies the updated signal immediately.

### is_lane_green(tls_id, lane_id)
Checks whether the TLS currently shows green for a given lane.

**Input:**
- `tls_id` (str): Traffic light identifier.
- `lane_id` (str): Lane to query.

**Output:**
- `True` if green, otherwise `False`.

**What it does:** Matches TLS phase indices to controlled lanes and checks whether the current signal color is green.

### spat_message_log(message)
Logs a SPaT (Signal Phase and Timing) message.

**Input:**
- `message` (str): Message text.

**Output:**
- `None`

**What it does:** Logs the provided message with simulation timestamp at INFO level.

### dynamic_tls(tls_id)
Main rule-based adaptive traffic signal controller.

**Input:**
- `tls_id` (str): Traffic light identifier.

**Output:**
- `None`

**What it does:** This method is executed once per simulation step.
1. Implements dynamic traffic-light control logic:
2. Restores the default TLS program when override expires.
3. Detects approaching vehicles grouped by lane.
4. Extends green time when a vehicle is close.
5. Switches to green if only one direction has approaching vehicles.
6. Resolves lane imbalance by temporarily giving green to lightly used lanes.

### _parse_rl_action(action)
Parses and normalizes the reinforcement learning action vector.

**Input:**
- `action`: (array-like): Action provided by the RL agent.

**Output:**
- `(alpha, beta)` (tuple of floats):
   - `alpha`: Detection range scaling factor.
   - `beta`: Green phase extention time scaling factor.

**What it does:**
- Flattens the action input if it is a NumPy array
- Extracts the second and third elements of the action vector
- Handles edge cases where the action vector is shorter than expected
- Falls back to default neutral values (0.5, 0.5) if the input is invalid or missing


### take_action(action)
Executes a traffic-light update step.

**Input:**
- `action`: RL action.

**Output:**
- `None`

**What it does:**
- Clears the per-step traffic light event buffer.
- Retrieves all traffic light IDs in the simulation.
- If RL mode is enabled:
   - Parses the action vector into control parameters.
   - Updates the vehicle detection range
   - Updates the green phase extension duration.
- Loops through all TLS systems and applies dynamic_tls() to each.
- Logs aggregated traffic light events for the current simulation step.

### get_observation
Constructs the current observation vector for the reinforcement learning agent.

**Input:** `None`

**Output:**
- `observation` (np.ndarray): A flattened observation vector representing the current state of all traffic lights.

**What it does:**
- Retrieves all traffic light IDs in the simulation.
- For each traffic light:
   - Groups controlled lanes by cardinal direction (N, S, E, W)
   - Computes the total number of vehicles per direction
   - Clamps queue lengths to a predefined maximum
   - Computes the remaining time until the next phase switch
- Appends all values into a single flattened observation vector
- Observation structure per traffic light:
   - 4 values: Queue lengths for controlled lanes (clamped)
   - 1 value: Time until the next traffic light phase switch

### calculate_reward
Computes the reinforcement learning reward based on traffic efficiency and control behavior.

**Input:** `None`

**Output:**
- `reward` (float): Calculated reward value.

**What it does:**
- Iterates over all traffic lights in the simulation
- Measures traffic performance metrics including:
   - Vehicle waiting time
   - Queue lengths
   - Traffic light phase switches
   - Throughput of moving vehicles
- Applies bonuses and penalties to encourage efficient traffic control
- Aggregates all components into a single scalar reward

**Reward components:**
- `Penalties:`
   - Average waiting time (avg_waiting): Mean waiting time of vehicles within the detection range
   - Total queue length (total_queue): Number of halting vehicles across all lanes
   - Phase switches (switches): Penalizes excessive signal switching
   - Time penalty: Small increasing penalty proportional to simulation time
- `Bonuses:`
   - Queue prioritization bonus: Rewards serving lanes with the largest queues
   - Throughput bonus: Rewards vehicles moving at non-zero speeds
   - Efficiency bonus: Rewards serving green lanes with detected vehicles
   - Parameter efficiency bonus: Rewards reasonable values of detection range and extension time

### get_feature_name
Returns the name of the feature module.

**Input:** `None`

**Output:**
- `feature_name` (str): Name of the feature.

**What it does:** Provides a human-readable identifier for the feature. Used for logging, debugging, and feature management within the framework.

### _log_tls_events()
Logs and/or displays traffic light events collected during the current simulation step.

**Input:** `None`

**Output:** `None`

**What it does:**
- Checks if there are any traffic light events in the per-step buffer _tls_log_events.
- If running in a terminal (sys.stdout.isatty()):
   - Displays a concise summary in the terminal using terminal_display
   - Shows the total number of events and the latest event snippet
- If not running in a terminal:
   - Logs all verbose events using the feature-level logger (logger.info)
- Provides a unified mechanism to visualize or log traffic light activity per step, supporting both TTY and non-TTY environments

### feature_step
Logs a debug message at each simulation step for monitoring feature parameters.

**Input:** `None`

**Output:** `None`

**What it does:**
- Retrieves the current values of the RL parameters:
   - detection_range (distance for detecting approaching vehicles)
   - extend_time (green phase extension duration)
- Logs a debug message containing the feature name and current parameter values
- Provides step-by-step visibility into the feature’s internal state for debugging or analysis

### feature_reset
Resets the internal state of the DynamicTLS feature at the start of a simulation or episode.

**Input:** `None`

**Output:** `None`

**What it does:**
- Clears the per-step traffic light event buffer (_tls_log_events)
- Records the current phase for all traffic lights in _last_phases
- Resets the cumulative phase time counter (phase_time)
- Clears lane-specific and traffic light-specific green timing records (lane_last_green and tls_last_switch)
- Logs a debug message indicating that the feature has been reset

# Priority Corridor Feature

### Constructor
Initializes the `PriorityCorridorFeature` responsible for giving way to emergency vehicles and optimizing their passage through intersections using rule-based or RL-based control.

**Input:**
- `feature_name` (str): Name of the feature (default: `"PriorityCorridorFeature"`).
- `enabled` (bool): Whether the feature is active (default: `True`).
- `rl_mode` (bool): Whether Reinforcement Learning mode is enabled (default: `False`).

**Output:** `None` (initializes object state)

**What it does:**  
Sets up internal state for both rule-based and RL modes. 
- **Rule-based state:** Caches emergency vehicle IDs, total yields, and per-step logs.
- **RL state:** Manages tracking for vehicle counts, TLS status, and reward calculation metrics.
- **Constants:**
    - `PRIORITY_TYPE`: Default vehicle type for emergency (`"emergency"`).
    - `RETURN_DISTANCE`: Distance (200m) after which yielding vehicles return to normal behavior.
    - `LANE_FREE_DIST`: Minimum clearance (8m) required for a safe yield merge.

### get_observation_space
Defines the observation space for the priority feature.

**Input:** `None`

**Output:**
- `gym.spaces.Box`: A 6-dimensional vector (RL mode) or a 3-dimensional vector (Rule-based mode).

**What it does:**
In RL mode, it returns a Box space representing:
1. `Priority Count`: Number of active priority vehicles.
2. `Average Waiting Time`: Mean waiting time of priority vehicles.
3. `Minimum Distance`: Proximity of the closest priority vehicle to the intersection.
4. `TLS Phase`: Current phase index of the controlled traffic light.
5. `Priority Queue`: Number of priority vehicles currently waiting.
6. `Other Queue`: Number of non-priority vehicles waiting.

### get_action_space
Defines the available actions for the priority feature.

**Input:** `None`

**Output:**
- `gym.spaces.Discrete`: 3 possible actions (RL mode) or 2 (Rule-based mode).

**What it does:**
Makes one TraCI pass to cache positions and edges for all vehicles and updates `_emergency_vehicle_ids` based on `PRIORITY_TYPE`.

### get_observation
Collects the current state of the priority corridor for the RL agent.

**Input:** `None`

**Output:**
- `np.ndarray`: A normalized 6-element observation vector.

**What it does:**
- Retrieves all active priority vehicles and computes their count, average wait time, and minimum distance.
- Queries the traffic light for its current phase and queue lengths.
- Normalizes all values (e.g., dividing distance by `max_distance`, count by 10) and clips them between 0.0 and 1.0.

### calculate_reward
Computes the reward signal for the RL agent based on priority vehicle efficiency.

**Input:** `None`

**Output:**
- `float`: Total step reward.

**What it does:**
- **Bonuses:** `+10.0` for every priority vehicle that successfully clears the intersection.
- **Penalties:**
    - `-0.5` per step for every priority vehicle that is stopped (speed < 0.1).
    - `-0.05` per unit of total queue length at the intersection.
    - `-1.0` for switching traffic light phases (encourages stability).
Counts vehicles per lane on the given edge using `traci.lane.getLastStepVehicleIDs` and returns the lane index with the fewest vehicles. If TraCI fails, falls back to lane `0`.

### _cache_positions_and_detect_emergencies
**Input:** `vehicle_ids` (Iterable[str])

**Output:** `positions` (dict), `edges` (dict), `edge_to_vehicle_ids` (dict)

**What it does:** Performs a single batch TraCI pass to cache vehicle state and identifies new emergency vehicles based on `PRIORITY_TYPE`.

### _choose_best_lane_for_emergency(edge_id)
**Input:** `edge_id` (str)

**Output:** `least_used_lane` (int)

**What it does:** Analyzes all lanes on the given edge and returns the index of the lane with the fewest vehicles, ensuring the priority vehicle has the clearest path.
**What it does:**
Checks if the target lane has enough local space for a safe merge by comparing the merging vehicle’s `(x, y)` position to other vehicles in that lane and enforcing a minimum distance `LANE_FREE_DIST` in both axes.

### _lane_is_free_enough(edge_id, lane_index, positions, vehicle_id)
**Input:** `edge_id`, `lane_index`, `positions`, `vehicle_id`

**Output:** `bool`

**What it does:** Validates if a target lane has sufficient longitudinal and lateral clearance (`LANE_FREE_DIST`) for a vehicle to merge safely.

### _log_priority_events()
**Input:** `None`

**Output:** `None`

**What it does:** Aggregates and displays yield events. In interactive terminals, it shows a "live" summary via `terminal_display`; otherwise, it writes verbose logs to the system logger.

### take_action(action)
**Input:** `action`: Action decided by the agent or runner.

**Output:** `None`

**What it does:**
1. **Always:** Executes rule-based lane yielding logic (corridor creation).
2. **If RL Mode:** Parses the provided action (0, 1, or 2) and applies it to the controlled traffic light using `traci.trafficlight.setPhase`.

### _perform_lane_yielding()
**Input:** `None`
Aggregates the per-step yield events:
- **TTY (interactive terminal):** shows a compact line via `terminal_display` with total yields and the latest short event.
- **Non-TTY (piped to file):** writes each verbose event string to the logger at `INFO` level.

**Output:** `None`

**What it does:** The core logic for corridor creation. It iterates through vehicles on the same edge as emergency vehicles and commands them to change lanes if they are in the path and have a safe adjacent lane to move into.

### feature_reset
**Input:** `None`

**Output:** `None`

**What it does:** Resets all internal counters, caches, and identifies the first available traffic light ID to control in RL mode.

### _get_priority_vehicles()
**Input:** `None`

**Output:** `list[str]`: IDs of vehicles matching priority types (`emergency`, `ambulance`, `police`, `fire`).

### _get_total_queue_length()
**Input:** `None`

**Output:** `int`: Total number of halting vehicles across all lanes controlled by the traffic light.

### get_feature_name
Provides the human-readable name of the feature for logging.
**What it does:**
Implements the priority corridor behavior each simulation step:

1. Reads all vehicle IDs and caches `positions`, `edges`, and `edge_to_vehicle_ids`.
2. Filters active emergency vehicles and, for each:
   - Finds its edge and the least-used lane via `_choose_best_lane_for_emergency`.
3. For every other vehicle on that edge:
   - Skips vehicles behind the emergency vehicle using `traci.vehicle.getLanePosition` (lane progression, not `(x, y)`).
   - Uses the squared distance to the emergency vehicle and `RETURN_DISTANCE` to decide when to restore default `laneChangeMode`.
   - Only processes vehicles in the lane chosen for the emergency vehicle.
   - Skips stopped vehicles (speed `< 0.1`).
   - Builds a small list of adjacent target lanes (left/right) and checks them with `_lane_is_free_enough`.
   - If `traci.vehicle.couldChangeLane` allows it, performs a short `changeLane` into a safe adjacent lane, increments the total yield counter, and records a verbose + short log entry.
4. Respects `MAX_BULK_COMMANDS_PER_STEP` to avoid flooding TraCI.
5. Calls `_log_priority_events()` once at the end of the step.

# Terminal Display

### Constructor
```python
def __init__(self, keys=None, logger_obj=None)
```

**Input:**
- `keys` (list, optional): Initial list of display keys for tracking multiple output lines. (default: `[]`)
- `logger_obj` (logging.Logger, optional): Logger instance for non-interactive output. (default: module-level logger)

**Output:** `None` (initializes object state)

**What it does:** Sets up the display manager with optional initial keys and logger. Initializes internal state tracking for values, logged output, and terminal initialization status.

### update(key, text)

**Input:**
- `key` (str): Display key/identifier for a specific output line
- `text` (str): The text to display for this key

**Output:** `None`

**What it does:** Updates the value for a given key. If the key doesn't exist, it's automatically added to the display. Stores the text for rendering on the next `render()` call.

### render()

**Input:** `None`

**Output:** `None`

**What it does:**
- **In TTY (interactive terminal):** Moves cursor up to overwrite previous lines in-place, creating a "live update" effect without scrolling
- **In Non-TTY (piped output, file logging):** Emits only changed values as INFO log messages to avoid spam
- On first call, initializes the display by printing all current lines
- On subsequent calls, detects changes and updates accordingly

### finish()

**Input:** `None`

**Output:** `None`

**What it does:** Cleans up the interactive display by moving the cursor to the line after the last display line and printing a newline. Resets the initialization state for potential future reuse.

### Module-Level Singleton

```python
terminal_display = TerminalDisplay(keys=["ENV"])
```

A module-level singleton instance is provided for global use across the application. Import and use directly:

```python
from terminal_display import terminal_display

terminal_display.update("ENV", "Simulation running...")
terminal_display.render()
```

# Progress Bar

### Constructor
```python
def __init__(self, logger: logging.Logger)
```

**Input:**
- `logger` (logging.Logger): Logger instance for recording progress events

**Output:** `None` (initializes object state)

**What it does:** Initializes the progress bar with ANSI color codes for terminal display, sets up tracking variables for trips and progress, and stores the logger reference.

### load_trip_paths()

**Input:** `None`

**Output:** `None`

**What it does:** Scans the `config/` directory recursively for all `.trips.xml` files and stores their paths in `self.file_paths` for later processing.

### count_total_trips()

**Input:** `None`

**Output:** `None`

**What it does:** Iterates through all trip files found by `load_trip_paths()`, counts occurrences of `<trip ` tags in each file, and stores the total count in `self.total_trips`.

### update(step)

**Input:**
- `step` (int): Number of trips/steps to increment the progress counter by (default: 1)

**Output:** `None`

**What it does:** Increments the current progress counter and calls `display()` to render the updated progress bar.

### return_progress_color(percent)

**Input:**
- `percent` (float): Progress percentage (0-100)

**Output:**
- (str): ANSI color code string

**What it does:** Returns color based on progress level:
- RED (< 50%)
- YELLOW (50-80%)
- GREEN (≥ 80%)

### display_string(current, end, info, steps)

**Input:**
- `current` (int): Current progress value (default: 0)
- `end` (int): End/total value (default: 1)
- `info` (str): Additional info text to display (default: '')
- `steps` (bool): If True, display as "Steps"; if False, display as "Arrived vehicles" (default: False)

**Output:**
- (str): Formatted progress bar string

**What it does:** Generates a progress bar display string with the current progress percentage and optional info text.

### display_string_bar(current, end, info)

**Input:**
- `current` (int): Current progress value (default: 0)
- `end` (int): End/total value (default: 1)
- `info` (str): Additional info text to display (default: '')

**Output:**
- (str): Formatted progress bar with filled and empty segments

**What it does:**
- Clears the line to prevent display artifacts
- Calculates progress percentage
- Determines color based on percentage
- Creates a 50-character bar with filled (█) and empty (-) segments
- Returns formatted string with color-coded bar and percentage

# RL Tester
**Input**
- **Command-line flags**
  - `--tls` (bool): Run Traffic Light System RL
  - `--bsm` (bool): Run Basic Safety Message RL
  - `--gui` (bool): Enable SUMO GUI

- **Internal parameters**
  - `sumo_config` (str): `config/simulation.sumocfg`

---

**Output**
- **Simulation metrics**
  - Total accumulated reward
  - Total number of steps executed

- **Process exit codes**
  - `68`: No feature selected
  - `67`: Multiple features selected
  - `56`: Required model not found

---

**What it does**
- Validates CLI arguments to ensure exactly one RL feature (TLS or BSM) is selected
- Initializes the SUMO reinforcement learning environment with the selected feature enabled
- Optionally enables the SUMO GUI for visual inspection
- Loads the corresponding pre-trained PPO model (`tls_feature_model` or `bsm_feature_model`)
- Terminates execution if the required model file is missing
- Runs a deterministic PPO-controlled simulation until the episode terminates
- Tracks cumulative reward and total step count during execution
- Closes the environment and prints final simulation metrics

---

# RL Trainee
**Input**
- **Command-line flags**
  - `--tls` (bool): Enable Traffic Light System RL training
  - `--bsm` (bool): Enable Basic Safety Message RL training
  - `--priority` (bool): Enable Priority-based RL training

- **Internal parameters**
  - `stop_time` (datetime): Absolute wall-clock time at which training must stop
  - `sumo_config` (str): Path to the SUMO configuration file

---

**Output**
- **Trained PPO model**
  - Saved as: `<feature>_feature_model`
- **Process exit codes**
  - `68`: No feature specified
  - `67`: Multiple features specified

---

**What it does**
- Disables CUDA to enforce CPU-only execution
- Validates CLI arguments to ensure exactly one SUMO feature is selected
- Initializes a SUMO-based reinforcement learning environment with only the chosen feature enabled
- Trains a PPO agent (Stable-Baselines3) using fixed hyperparameters and an unbounded timestep budget
- Uses a custom `StopAtTimeCallback` to terminate training when a predefined wall-clock time is reached
- Saves the trained model for the selected feature
- Performs a short, deterministic post-training evaluation rollout (up to 1000 steps) to validate the policy

---

# Simulation Runner Class

### Constructor
```python
def __init__(self, config_path, sumo_env, steps, **kwargs)
```

**Input:**
- `config_path` (str): Path to SUMO configuration file
- `sumo_env` (BaseSumoEnvironment or class): Pre-instantiated environment or environment class
- `steps` (int, optional): Number of simulation steps to run
- `**kwargs`: Additional keyword arguments passed to the environment

**Output:** `None` (initializes object state)

**What it does:** Sets up the simulation runner with either a provided environment instance or creates one from the config. Stores the step limit for later use in simulation execution.

### run_manual_feature_test()

**Input:** `None`

**Output:** `None`

**What it does:**
- Logs which features are active
- Resets the environment
- Loops through simulation steps, taking random actions
- Updates and renders display at each step
- Automatically resets if episode terminates
- Closes environment when done

### test_specific_feature(feature_name)

**Input:**
- `feature_name` (str): Name of the feature to test in isolation

**Output:** `None`

**What it does:**
- Logs isolated feature testing mode
- Resets the environment
- Steps through simulation with custom actions for the feature
- Logs debug information per step
- Handles episode resets
- Closes environment when done

### _get_feature_specific_action(feature_name, step)

**Input:**
- `feature_name` (str): Name of the feature being tested
- `step` (int): Current simulation step number

**Output:** Action compatible with the environment's action space

**What it does:** Generates a feature-specific action for testing. Currently returns a random action from the action space; can be overridden for custom feature-specific logic.

### run_until_end()

**Input:** `None`

**Output:** `None`

**What it does:**
- Steps through simulation until no more vehicles are expected
- Updates terminal display with current time and vehicle count at each step
- Handles FatalTraCIError exceptions gracefully
- Calls `terminal_display.finish()` to clean up display when done
- Logs completion message

### run_with_steps()

**Input:** `None`

**Output:** `None`

**What it does:**
- Steps through exactly `self.simulation_steps` iterations
- Updates terminal display with current time and vehicle count at each step
- Exits loop after configured step count

### start_simulation()

**Input:** `None`

**Output:** `None`

**What it does:**
- Resets the environment to initialize SUMO
- Decides between `run_with_steps()` (if steps configured) or `run_until_end()` (if open-ended)
- Closes the environment after execution completes

### parse_arguments()

**Input:** `None`

**Output:** `argparse.Namespace` containing parsed command-line arguments

**What it does:** Parses command-line arguments for:
- `--steps`: Number of simulation steps
- `--gui`: Enable SUMO GUI
- `--bsm`: Enable Basic Safety Message feature
- `--tls`: Enable Traffic Light System feature
- `--priority`: Enable priority vehicle handling
- `--reroute`: Enable dynamic rerouting
- `--test-all`: Test all features with manual control

### main()

**Input:** `None` (reads from command-line arguments)

**Output:** `None`

**What it does:**
1. Parses command-line arguments
2. Constructs path to SUMO config file
3. Creates BaseSumoEnvironment with enabled features
4. Creates SimulationRunner instance
5. Determines execution mode based on enabled features:
   - `--test-all`: Run manual feature test mode
   - Single feature enabled: Run isolated feature test
   - Multiple features: Run manual feature test mode
   - No features: Run standard simulation
6. Executes the chosen simulation mode

# Main

Entry point for the V2X simulation framework. Provides a command-line interface to launch different simulation runners.

### main()

**Input:** `None` (reads from command-line arguments)

**Output:** `None`

**What it does:**
1. Creates an argument parser with mutually exclusive group for runner selection
2. Defines four available commands:
   - `--runner`: Launch the simulation runner
   - `--collect`: Launch the data collector runner
   - `--rl`: Launch the RL training module (rl_trainee)
   - `--rltest`: Launch the RL testing module (rl_tester)
3. Parses the arguments and validates that one command is selected
4. Maps the selected command to its corresponding module path
5. Adds the `src/` directory to the Python path
6. Uses `runpy.run_module()` to execute the selected module, passing through any remaining arguments

**Command Mapping:**
- `--runner` → `runners.simulation_runner`
- `--collect` → `runners.collector_runner`
- `--rl` → `runners.rl_trainee`
- `--rltest` → `runners.rl_tester`

<br>
<br>
<br>

# Usage and Examples

**Run simulation with specified steps:**
```bash
python main.py --runner --steps 100
```

**Run data collection:**
```bash
python main.py --collect
```

**Train RL model:**
```bash
python main.py --rl
```

**Test trained RL model:**
```bash
python main.py --rltest
```

**Enable features with simulation:**
```bash
python main.py --runner --bsm --tls --gui
```
