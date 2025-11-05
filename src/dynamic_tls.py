import traci
import time

# Distance threshold between an incoming vehicle and a traffic light (in meters)
detection_range = 50


# Time by which the green light will be exteded (in seconds)
extend_time = 5

# Stores times at which tls overrides were done in the simulation
tls_override_times = {}

# Returns distance from vehivle to tls
def get_distance_to_tls(v_id, tls_id):
    try:
        dist = traci.vehicle.getDrivingDistance(v_id,tls_id)
        return dist
    except:
        return float("inf")

# Groups appraching vehicles by lane within the detection range
def get_approaching_vehicles_by_lane(tls_id, detection_range):
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    approaching = {}

    for lane_id in lanes:
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        lane_length = traci.lane.getLength(lane_id)
        near_tls = []

        for v_id in vehicle_ids:
            vehicle_pos = traci.vehicle.getLanePosition(v_id)
            if lane_length - vehicle_pos < detection_range:
                near_tls.append(v_id)
        
        if near_tls:
            approaching[lane_id] = near_tls

    return approaching

# Returns all lanes belonging to a strret assuming consisntent naming(eg. 526477801#1_0,526477801#1_1)
def get_lanes_on_same_street(tls_id, lane_id):
    edge_id = traci.lane.getEdgeID(lane_id)
    all_lanes = traci.trafficlight.getControlledLanes(tls_id)
    same_street_lanes = [lane for lane in all_lanes if traci.lane.getEdgeID(lane) == edge_id]
    return same_street_lanes

#Sets TLS light for named vehicle to green
def set_tls_green_for_vehicle(tls_id, v_id):
    lane_id = traci.vehicle.getLaneID(v_id)
    street_lanes = get_lanes_on_same_street(tls_id,lane_id)
    tls_state = list(traci.trafficlight.getRedYellowGreenState(tls_id))
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    for i, links in enumerate(controlled_links):
        lane_found = any(link[0] in street_lanes for link in links)
        tls_state[i] = 'G' if lane_found else 'r'

    traci.trafficlight.setRedYellowGreenState(tls_id,''.join(tls_state))

# Checks if the TLS light for the named lane is green
def is_lane_green(tls_id,lane_id):
    tls_state = traci.trafficlight.getRedYellowGreenState(tls_id)
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    for i, links in enumerate(controlled_links):
        if lane_id in [link[0] for link in links]:
            if tls_state[i].lower() != 'g':
                return False
    return True
    
def is_emergency_vehicle(v_id):
    return traci.vehicletype.getVehicleClass(traci.vehicle.getTypeID(v_id)) == "emergency"

# Prints SPaT (Signal Phase and Timing) messages to console
def spat_message_log(message):
    timestamp = traci.simulation.getTime()
    log_message = f"[{timestamp:.1f}s] {message}"
    print (log_message)

# Main dynamic TLS control function:
# - detects vehicles approaching intersections
# - prioritizes energency vehicles
# - extends geern lights dynamically
# - switches light to green if only one direction has vehicles
# - grants green light to fewer cars that are waiting for a lot of cars to pass
# - restores default TLS program after manual overrides
def dynamic_tls_control(tls_id):
    global tls_override_times
    current_time = traci.simulation.getTime()
    tls_lanes = traci.trafficlight.getControlledLanes(tls_id)
    vehicle_list = traci.vehicle.getIDList()

    # Checks if an override is already active and reverts to normal if time expired
    if tls_id in tls_override_times:
        if current_time - tls_override_times[tls_id] >= extend_time:
            traci.trafficlight.setProgram(tls_id,"0")
            spat_message_log(f"TLS {tls_id} returning to normal program")
            del tls_override_times[tls_id]

    for v_id in vehicle_list:
        lane_id = traci.vehicle.getLaneID(v_id)
        if lane_id not in tls_lanes:
            continue

        distance_to_tls = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(v_id)

        if distance_to_tls < detection_range * 2:
            # Case 1: Emergency vehicle
            if is_emergency_vehicle(v_id):
                set_tls_green_for_vehicle(tls_id, v_id)
                traci.trafficlight.setPhaseDuration(tls_id, extend_time * 2)
                tls_override_times[tls_id] = current_time
                spat_message_log(f"\033[91mEmergency vehicle {v_id} detected near {tls_id}, forcing GREEN\033[0m")
                return

            # Case 2: Normal vehicles
            if distance_to_tls < detection_range:

                remaining = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
            
                # Case 2.1: Extend already green light
                if is_lane_green(tls_id, lane_id) and remaining < extend_time:
                    traci.trafficlight.setPhaseDuration(tls_id, extend_time)
                    spat_message_log(f"Vehicle {v_id} approaching {tls_id}, extending GREEN for {extend_time}s.")
                    return
                
                # Case 2.2: Turn green if only one lane is approaching
                approaching = get_approaching_vehicles_by_lane(tls_id, detection_range)

                if len(approaching) == 1 and lane_id in approaching:
                    if not is_lane_green(tls_id, lane_id):
                        set_tls_green_for_vehicle(tls_id, v_id)
                        traci.trafficlight.setPhaseDuration(tls_id, extend_time)
                        tls_override_times[tls_id] = current_time
                        spat_message_log(f"Only vehicles on lane {lane_id} near {tls_id}, switching to GREEN")
                    return
                
                # Case 2.3: Turn light green if there is only one vehicle waiting for a lot of vehicles to pass
                edge_counts = {edge: len(v_list) for edge, v_list in approaching.items()}

                if edge_counts:
                    max_edge = max(edge_counts, key = edge_counts.get)
                    min_edge = min(edge_counts, key = edge_counts.get)

                    max_count = edge_counts[max_edge]
                    min_count = edge_counts[min_edge]

                    if (min_count > 0 and min_count <= 3) and max_count - min_count > 5:
                        v_id = approaching[min_edge][0]
                        if not is_lane_green(tls_id,lane_id):
                            set_tls_green_for_vehicle(tls_id,v_id)
                            traci.trafficlight.setPhaseDuration(tls_id, extend_time)
                            tls_override_times[tls_id] = traci.simulation.getTime()
                            spat_message_log(f"\033[93mImbalance detected at {tls_id}, granting short green for lane {min_edge}\033[0m")
    return