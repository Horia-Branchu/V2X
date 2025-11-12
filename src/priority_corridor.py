from __future__ import annotations
from typing import Iterable
import traci

try:
    from simulation_runner import SimulationRunner
except Exception as _e:
    raise RuntimeError("priority_corridor.py expects simulation_runner.py in PYTHONPATH") from _e

PRIORITY_PREFIXES: tuple[str, ...] = ("amb", "fire", "pol")
TRIGGER_DISTANCE: float = 35.0          # meters
SLOWDOWN_FACTOR: float = 0.35           # 35% of current speed while yielding
YIELD_DURATION_STEPS: int = 3           # seconds to keep lane change order

MIN_SPEED_AFTER_SLOWDOWN: float = 0.0   # allow full stop
MAX_BULK_COMMANDS_PER_STEP: int = 500   # avoid sending too many TraCI cmds


def _is_priority(veh_id: str) -> bool:
    v = veh_id.lower()
    return any(v.startswith(p.lower()) for p in PRIORITY_PREFIXES)


def _euclid2(p1, p2) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


try:
    from simulation_runner import RunnerWithCollector as _BaseRunner
except Exception:
    _BaseRunner = SimulationRunner


class PriorityCorridorRunner(_BaseRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._priority_speed_mode = 0
        self._priority_lc_mode = 0

    def _enforce_priority_corridors(self):
        veh_ids: Iterable[str] = traci.vehicle.getIDList()
        priors = [vid for vid in veh_ids if _is_priority(vid)]
        if not priors:
            return

        # Cache positions and edges to avoid repeated calls
        pos = {}
        edge = {}
        for vid in veh_ids:
            try:
                pos[vid] = traci.vehicle.getPosition(vid)
                edge[vid] = traci.vehicle.getRoadID(vid)
            except traci.TraCIException:
                continue

        # Rightmost - index 0 in sumo
        cmds_sent = 0
        for p in priors:
            if p not in pos or p not in edge:
                continue

            # Make priority vehicles less constrained so they can overtake
            try:
                traci.vehicle.setSpeedMode(p, self._priority_speed_mode)
                traci.vehicle.setLaneChangeMode(p, self._priority_lc_mode)
            except traci.TraCIException:
                pass

            p_edge = edge[p]
            p_pos = pos[p]

            # Consider only cars on the same edge as the priority car
            for v in veh_ids:
                if v == p or _is_priority(v):
                    continue
                if v not in pos or v not in edge:
                    continue
                if edge[v] != p_edge:
                    continue

                # Distance gate
                if _euclid2(pos[v], p_pos) > (TRIGGER_DISTANCE * TRIGGER_DISTANCE):
                    continue

                # Issue yield directives: change to lane 0 and slow down
                try:
                    traci.vehicle.changeLane(v, 0, YIELD_DURATION_STEPS)
                    cur = traci.vehicle.getSpeed(v)
                    new_speed = max(MIN_SPEED_AFTER_SLOWDOWN, cur * SLOWDOWN_FACTOR)
                    traci.vehicle.setSpeedMode(v, 0)
                    traci.vehicle.setSpeed(v, new_speed)
                    cmds_sent += 1
                    if cmds_sent >= MAX_BULK_COMMANDS_PER_STEP:
                        return
                except traci.TraCIException:
                    continue

    def run_until_end(self):

        traci.simulationStep()
        if hasattr(self, "collector"):
            try:
                current_time = traci.simulation.getTime()
                self.collector.collect(current_time)
            except Exception:
                pass
        self._enforce_priority_corridors()

        try:
            while traci.simulation.getMinExpectedNumber() != 0:
                traci.simulationStep()
                if hasattr(self, "collector"):
                    try:
                        current_time = traci.simulation.getTime()
                        self.collector.collect(current_time)
                    except Exception:
                        pass
                self._enforce_priority_corridors()
        except traci.exceptions.FatalTraCIError as e:
            try:
                import logging
                logging.getLogger(__name__).error("Fatal TraCI error in PriorityCorridorRunner: %s", e)
            except Exception:
                pass

    def run_with_steps(self):
        steps = getattr(self, "simulation_steps", None)
        if not isinstance(steps, int) or steps <= 0:
            return self.run_until_end()

        for _ in range(steps):
            traci.simulationStep()
            if hasattr(self, "collector"):
                try:
                    current_time = traci.simulation.getTime()
                    self.collector.collect(current_time)
                except Exception:
                    pass
            self._enforce_priority_corridors()
