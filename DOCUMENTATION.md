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

**Output:**
`None` (initializes internal state)

**What it does:**
Creates and configures the dynamic traffic-light-control feature, defining observation/action sizes, vehicle detection range, green-light extension duration, and bookkeeping structures for temporary TLS overrides.

### get_approaching_vehicles_by_lane(tls_id)
Groups approaching vehicles per lane within the configured detection range.

**Input:**
- `tls_id` (str): Traffic light identifier.

**Output:**
- `approaching` (Dictionary): { lane_id: [vehicle_ids...] }

**What it does:**
- For each controlled lane:
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

**What it does:**
- Matches edges between the reference lane and all lanes controlled by the TLS, returning those on the same road.

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

**What it does:**
- Matches TLS phase indices to controlled lanes and checks whether the current signal color is green.

### spat_message_log(message)
Logs a SPaT (Signal Phase and Timing) message.

**Input:**
- `message` (str): Message text.

**Output:**
- `None`

**What it does:**
- Logs the provided message with simulation timestamp at INFO level.

### dynamic_tls(tls_id)
Main rule-based adaptive traffic signal controller.

**Input:**
- `tls_id` (str): Traffic light identifier.

**Output:**
- `None`

**What it does:**
1. Implements dynamic traffic-light control logic:
2. Restores the default TLS program when override expires.
3. Detects approaching vehicles grouped by lane.
4. Extends green time when a vehicle is close.
5. Switches to green if only one direction has approaching vehicles.
6. Resolves lane imbalance by temporarily giving green to lightly used lanes.

- This method is executed once per simulation step.

### take_action(action)
Executes a traffic-light update step.

**Input:**
- `action`: RL action (currently unused).

**Output:**
- `None`

**What it does:**
- Loops through all TLS systems and applies dynamic_tls() to each.

# Simulation Runner Class
