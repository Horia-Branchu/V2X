#!/bin/bash

python3 ~/sumo/tools/randomTrips.py   -n cluj_network.net.xml.gz   -p 50 -e 500  -o cluj_traffic.emergency.trips.xml --prefix="e1_" --trip-attributes="type=\"emergency\"" --weights-prefix edge_weights --allow-fringe &
python3 ~/sumo/tools/randomTrips.py   -n cluj_network.net.xml.gz   -p 0.28 -e 700  -o cluj_traffic.passenger.trips.xml --prefix="p1_" --weights-prefix edge_weights --allow-fringe &
python3 ~/sumo/tools/randomTrips.py   -n cluj_network.net.xml.gz   -p 0.28 -e 700  -o cluj_traffic.passenger2.trips.xml --prefix="p2_" --weights-prefix edge_weights --allow-fringe &
wait

# original attributes are -p 0.2 -e 500 for passenger and -p 50 -e 500 for emergency, keep this in
#mind when changing stuff here.