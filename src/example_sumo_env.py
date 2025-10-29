import traci
from base_sumo_env import BaseSumoEnvironment

class ExampleSumoEnviroment(BaseSumoEnvironment):
    """
        A simple example for a Sumo Enviroment implementation
    """
    def _take_action(self, action):
        # no actions needed for simple simulation running
        print(action) # remove for an actual implementation for the enviroment
        pass

    def _get_observation(self):
        # return basic simulation info as observation
        return {
            'time': traci.simulation.getTime(),
            'vehicle_count': traci.vehicle.getIDCount()
        }

    def _calculate_reward(self):
        # no reward computation for simple running
        return 0.0

    def _get_info(self):
        return {
            'step': self.current_step,
            'vehicles':traci.vehicle.getIDCount(),
            'time': traci.simulation.getTime()
        }

    def _scenario_reset(self):
        # no special reset logic needed
        pass

