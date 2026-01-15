# 🚦 V2X Simulation using SUMO & TraCI

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/Eclipse%20SUMO-1.24.0-orange?logo=eclipse&logoColor=white)](https://www.eclipse.org/sumo/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white)](LICENSE)

A **Vehicle-to-Everything (V2X)** traffic simulation built with **Eclipse SUMO** and **Python (TraCI)**.  
The project models connected-vehicle interactions and **adaptive traffic-light control** reacting to real-time traffic conditions in an urban environment.

---

## Overview

This simulation demonstrates how connected vehicles and intelligent traffic infrastructure can cooperate to improve road safety and efficiency.

The base network models the **city of Cluj-Napoca (Romania)**, featuring realistic road topology and intersections.  
However, **any other city or custom road network** can be simulated by providing your own `.sumocfg` configuration file.

Core concepts:

- Vehicles exchange **Basic Safety Messages (BSM)**.  
- **Traffic lights** adjust dynamically based on live traffic density and queue data.  
- Incidents, blocked roads or high congestions automatically trigger **rerouting** behaviors.

---

## ⚙️ Components

| Component | Role |
|------------|------|
| **Eclipse SUMO** | Microscopic traffic simulator |
| **NetEdit** | Network editor for roads, junctions, and signals |
| **Python + Sumolib** | Logic layer for V2X behavior and dynamic TLS |
| **Network** | City of Cluj-Napoca (configurable) |
| **SUMO Version** | 1.24.0 |
| **Python Version** | 3.x |

---

## Features

- 🗺️ Realistic network: **Cluj-Napoca traffic model**
- 📡 V2V communication (BSM-based proximity monitoring)
- 🚦 **Dynamic traffic-light control** reacting to congestion
- 🧾 Logging of TLS states, reroutes, and safety events
- 📊Post-simulation data analysis to compare standard driving vs. V2X-enhanced scenarios
- ⚙️ Easily switch to your own city or scenario by configuring your own sumo network in the config directory

---

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for details about the project structure and code modules.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---
