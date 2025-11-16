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

# Simulation Runner Class
