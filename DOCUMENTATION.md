# Documentation

## Project Structure
├── data <br>
├── config <br>
│   ├── network.net.xml <br>
│   ├── routes.rou.xml <br>
│   └── simulation.sumocfg <br>
├── src <br>
│   ├── agents <br>
│   │   └── ppo.py <br>
│   ├── [base_sumo_env.py](#base-sumo-environment-class) <br>
│   ├── default_sumo_env.py <br>
│   └── [simulation_runner.py](#simulation-runner-class) <br>


## Modules
# Base Sumo Environment Class

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

**Output:** `None` (initializes object state)

**What it does:** Sets up simulation parameters, initializes step counter, and builds SUMO command.

### _build_sumo_command

**Input:** `None`

**Output:** Returns the string to call sumo in the command line

**What it does:** It initializez the sumo base command based on the arguments (can be either with the GUI or in the command line).

### _check_unimplemented_features

**Input:** `None`

**Output:** `None`

**What it does:** It check for the unimplemented features and logs with the warning method the feature that is not implemented yet.

### reset
**Input:**
- `seed`: (int, optional): Random seed for reproducibility. (default: `None`)
- `options`:  (dict, optional): Additional reset options. (default `None`)

**Output:**
- `observation`: Initial environment observation after reset
- `info`: Dictionary containing auxiliary information

**What it does:** Closes any existing simulation, starts a new simulation instance, resets step counter, checks features, resets scenario, and returns initial observation data.

### stepon
**Input:**
- `action`: (): Individual action the algorithm should take at any given step.
  
**Output:**
- `observation`:
- `reward`:
- `terminated`:
- `truncated`:
- `info`:

**What it does:** Takes the custom action, moves on with the simulation, increases the counter then it collects the results (observation, reward, terminated, truncated, info) and then returns them.

### close
**Input:** `None`

**Output:** `None`

**What it does:**: It tries to close the Traci Module and if there is any GUI opened, it terminates the process based on the current OS running.

### parse_arguments
**Input:** `None`

**Output:** Output the object with all the arguments passed at executino in the command line.

**What it does:**: It maps the arguments given in the command line with the internal object. At the same time writes a custom message for each argument.

### _take_action
**Input:** None

**Output:** None

**What it does:**

### _get_observation
**Input:** None

**Output:** None

**What it does:**

### _calculate_reward
**Input:** None

**Output:** None

**What it does:**

### _is_terminated
**Input:** None

**Output:** None

**What it does:**

### _is_truncated
**Input:** None

**Output:** None

**What it does:**

### _get_info
**Input:** None

**Output:** None

**What it does:**


# Simulation Runner Class

### Constructor:
Starts the simulation and handles the V2X related implementations.

**Input**
- `config_path` (str): Path to SUMO configuration file
- `sumo_env` (class): Concrete implementation of BaseSumoEnvironment (optional, defaults to DefaultSumoEnvironment)
- `**kwargs`: Additional arguments passed to the environment

**Output:** `None` (initializes object state)

**What it does:** Creates a simulation environment instance, falling back to DefaultSumoEnvironment if none provided, and initializes the environment with the given configuration and arguments.

### run_until_end
**Input:** `None`

**Output:** `None`

**What it does:** Runs the simulation until it naturally ends by checking if there are no more expected vehicles and no active vehicles. Logs vehicle count at each step and handles TraCI fatal errors 'gracefully'.

### run_steps
**Input:**
- `num_steps` (int): Number of simulation steps to run

**Output:** `None`

**What it does:** Runs the simulation for a specified number of steps, logging the current time and vehicle count at each step.

### start_simulation
**Input:** `None`

**Output:** `None`

**What it does:** Resets the environment, runs the simulation either for a fixed number of steps (if simulation_steps is set) or until natural end, then closes the environment.

### main
**Input:** `None`

**Output:** `None`

**What it does:** Entry point for running the simulation, sets up the environment and starts the simulation process.