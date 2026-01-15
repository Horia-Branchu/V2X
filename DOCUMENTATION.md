# Documentation

## Project Structure
├── data <br>
├── config <br>
│   ├── network.net.xml <br>
│   ├── routes.rou.xml <br>
│   └── simulation.sumocfg <br>
├── src <br>
│   ├── agents <br>
│   │   └── [ppo.py](#ppo)<br>
│   ├── [base_sumo_env.py](#base-sumo-environment) <br>
│   ├── [base_v2x_feature.py](#base-v2x-feature) <br>
│   ├── [dynamic_tls.py](#dynamic-tls) <br>
│   ├── [priority_corridor.py](#priority-corridor-feature) <br>
│   ├── [terminal_display.py](#terminal-display) <br>
│   └── [simulation_runner.py](#simulation-runner-class) <br>


## Modules

# PPO
This script creates a SUMO environment, validates it against Gymnasium
standards, initializes a PPO model with specific hyperparameters (learning
rate: 0.0003, n_steps: 2048, batch_size: 64, n_epochs: 10), trains the model
for 100,000 timesteps, saves the trained model to disk, then tests it by
running 1000 simulation steps with deterministic actions while automatically
handling episode resets.


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
In RL mode, provides discrete actions:
- `0`: Keep current traffic light phase.
- `1`: Switch to the next traffic light phase.
- `2`: Force green for the priority corridor (Phase 0).

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

### _cache_positions_and_detect_emergencies
**Input:** `vehicle_ids` (Iterable[str])

**Output:** `positions` (dict), `edges` (dict), `edge_to_vehicle_ids` (dict)

**What it does:** Performs a single batch TraCI pass to cache vehicle state and identifies new emergency vehicles based on `PRIORITY_TYPE`.

### _choose_best_lane_for_emergency(edge_id)
**Input:** `edge_id` (str)

**Output:** `least_used_lane` (int)

**What it does:** Analyzes all lanes on the given edge and returns the index of the lane with the fewest vehicles, ensuring the priority vehicle has the clearest path.

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

## Usage Examples

**Run simulation without features for specified steps:**
```bash
python main.py --runner --steps 100
```

**Run simulation until vehicles are depleted with TLS enabled:**
```bash
python main.py --runner --tls
```

**Enable multiple features with GUI:**
```bash
python main.py --runner --bsm --tls --gui
```

**Test specific feature in isolation:**
```bash
python main.py --runner --bsm
```

**Test all features with manual control:**
```bash
python main.py --runner --test-all #(not to be used yest since not all features are implemented)
```
