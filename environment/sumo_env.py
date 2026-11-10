"""Single ego car longitudinal-control environment for a SUMO intersection."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import locate_sumo_home

SUMO_HOME = locate_sumo_home()
sys.path.insert(0, str(SUMO_HOME / 'tools'))
import traci


class IntersectionEnv(gym.Env):
    """Control one red ego car; SUMO's safety checks remain enabled.

    Features: speed, remaining distance, signal, gap, leader speed, relative speed.
    Traffic-light feature: 0=red, 0.5=yellow, 1=green or no upcoming light.
    The reward is an initial teaching baseline, not a validated safety metric.
    """
    metadata = {'render_modes': ['human']}

    DENSITY_SCALE = {'low': 0.5, 'medium': 1.0, 'high': 1.5}

    def __init__(self, render_mode=None, max_steps=300, traffic_density='medium', gui_delay_ms=500, gui_quit_on_end=False):
        super().__init__()
        if render_mode not in (None, 'human'):
            raise ValueError('render_mode must be None or human')
        self.render_mode = render_mode
        self.gui_delay_ms = gui_delay_ms
        self.gui_quit_on_end = gui_quit_on_end
        if traffic_density not in self.DENSITY_SCALE:
            raise ValueError('traffic_density 必须是 low、medium 或 high')
        self.traffic_density = traffic_density
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1], dtype=np.float32),
            high=np.ones(6, dtype=np.float32), dtype=np.float32)
        self.connection = None
        self.ego = 'ego'
        self.steps = 0
        self.last_distance = 0.0
        self.red_light_violation = False
        self.entered_junction = False
        self.signal_boundary_uncertain = False

    def _observation(self):
        speed = self.connection.vehicle.getSpeed(self.ego)
        lane_pos = self.connection.vehicle.getLanePosition(self.ego)
        lane_id = self.connection.vehicle.getLaneID(self.ego)
        remaining = max(0.0, self.connection.lane.getLength(lane_id) - lane_pos) if lane_id.startswith('north_in_') else 0.0
        lights = self.connection.vehicle.getNextTLS(self.ego)
        signal = 1.0
        if lights:
            state = lights[0][3].lower()
            signal = 0.0 if state == 'r' else 0.5 if state == 'y' else 1.0
            remaining = max(0.0, lights[0][2])
        leader = self.connection.vehicle.getLeader(self.ego, 100.0)
        gap, leader_speed = 100.0, speed
        if leader:
            gap = min(100.0, max(0.0, leader[1]))
            leader_speed = self.connection.vehicle.getSpeed(leader[0])
        observation = np.array([speed/13.89, min(remaining,250)/250, signal,
                                gap/100, leader_speed/13.89,
                                (leader_speed-speed)/13.89], dtype=np.float32)
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def _details(self):
        """Readable, unscaled values for inspecting the scene and actions."""
        vehicle = self.connection.vehicle
        speed = vehicle.getSpeed(self.ego)
        leader = vehicle.getLeader(self.ego, 100.0)
        lights = vehicle.getNextTLS(self.ego)
        return {
            'speed_m_s': speed,
            'lane_id': vehicle.getLaneID(self.ego),
            'position_m': vehicle.getLanePosition(self.ego),
            'distance_to_signal_m': max(0.0, lights[0][2]) if lights else None,
            'signal': lights[0][3] if lights else None,
            'leader_gap_m': max(0.0, leader[1]) if leader else None,
            'leader_speed_m_s': vehicle.getSpeed(leader[0]) if leader else None,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        density = (options or {}).get('traffic_density', self.traffic_density)
        if density not in self.DENSITY_SCALE:
            raise ValueError('traffic_density 必须是 low、medium 或 high')
        self.close()
        binary = SUMO_HOME / 'bin' / ('sumo-gui.exe' if self.render_mode == 'human' else 'sumo.exe')
        command = [str(binary), '-c', str(ROOT/'environment'/'network'/'intersection.sumocfg'),
                   '--seed', str(int(seed if seed is not None else 42)), '--no-step-log', 'true',
                   '--scale', str(self.DENSITY_SCALE[density])]
        if self.render_mode == 'human':
            command += ['--start', '--delay', str(self.gui_delay_ms)]
            if self.gui_quit_on_end:
                command += ['--quit-on-end']
        # Separate labels make it possible to create more than one environment sequentially.
        label = f'intersection_{id(self)}'
        traci.start(command, label=label)
        self.connection = traci.getConnection(label)
        self.connection.vehicle.add(self.ego, 'n_s', typeID='car', depart='now', departLane='best')
        self.connection.vehicle.setColor(self.ego, (255, 0, 0, 255))
        # Lane changing is disabled; SUMO retains collision and signal safety checks.
        self.connection.vehicle.setLaneChangeMode(self.ego, 0)
        self.steps = 0
        self.red_light_violation = False
        self.entered_junction = False
        self.signal_boundary_uncertain = False
        # Insertion may be delayed by traffic; fail clearly rather than run indefinitely.
        for _ in range(60):
            self.connection.simulationStep()
            if self.ego in self.connection.vehicle.getIDList():
                break
        else:
            self.close()
            raise RuntimeError('主车在 60 秒内未能进入路网')
        self.last_distance = self.connection.vehicle.getDistance(self.ego)
        return self._observation(), {'ego_id': self.ego, 'traffic_density': density,
                                     'traffic_scale': self.DENSITY_SCALE[density],
                                     'safety': 'SUMO default speed checks', **self._details()}

    def step(self, action):
        if self.connection is None:
            raise RuntimeError('请先调用 reset()')
        if not self.action_space.contains(action):
            raise ValueError(f'无效动作: {action}')
        acceleration = [-3.0, -1.0, 0.0, 1.0, 2.0][int(action)]
        before_speed = self.connection.vehicle.getSpeed(self.ego)
        before_road = self.connection.vehicle.getRoadID(self.ego)
        before_tls = self.connection.vehicle.getNextTLS(self.ego)
        before_signal = before_tls[0][3] if before_tls else None
        self.connection.vehicle.setAcceleration(self.ego, acceleration, 1.0)
        self.connection.simulationStep()
        self.steps += 1
        arrived = self.ego in self.connection.simulation.getArrivedIDList()
        collided = self.ego in self.connection.simulation.getCollidingVehiclesIDList()
        teleported = self.ego in self.connection.simulation.getStartingTeleportIDList()
        exists = self.ego in self.connection.vehicle.getIDList()
        after_road = self.connection.vehicle.getRoadID(self.ego) if exists else None
        # Road transition from inbound to internal/outbound is the stop-line crossing.
        crossing = bool(exists and before_road == 'north_in' and
                        (after_road.startswith(':junction_') or after_road == 'south_out'))
        after_signal = None
        if crossing:
            self.entered_junction = True
            if before_tls:
                tls_id, link_index = before_tls[0][:2]
                after_signal = self.connection.trafficlight.getRedYellowGreenState(tls_id)[link_index]
            self.signal_boundary_uncertain = before_signal is None or after_signal != before_signal
            self.red_light_violation = bool(not self.signal_boundary_uncertain and before_signal.lower() == 'r')
        passed = bool(exists and after_road == 'south_out' and self.entered_junction and not collided and not teleported)
        terminated = bool(passed or arrived or collided or teleported or self.red_light_violation or not exists)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        distance = self.connection.vehicle.getDistance(self.ego) if exists else self.last_distance
        progress = max(0.0, distance-self.last_distance)
        self.last_distance = distance
        reward = (.1*progress-.05-(.1 if acceleration <= -3 else 0)
                  -100*int(collided)-100*int(crossing and self.red_light_violation)
                  +50*int(passed and not self.red_light_violation and not self.signal_boundary_uncertain))
        observation = self._observation() if exists else np.zeros(6, dtype=np.float32)
        info = {'arrived': arrived, 'passed_intersection': passed, 'collision': collided,
                'teleported': teleported, 'ego_present': exists,
                'progress_m': progress, 'requested_acceleration': acceleration,
                'actual_acceleration_m_s2': (self.connection.vehicle.getSpeed(self.ego)-before_speed) if exists else None,
                'crossed_stop_line': crossing, 'signal_before_crossing': before_signal if crossing else None,
                'signal_after_crossing': after_signal,
                'signal_boundary_uncertain': self.signal_boundary_uncertain,
                'red_light_violation': None if self.signal_boundary_uncertain else self.red_light_violation,
                'safety_interventions': None}
        if exists:
            info.update(self._details())
        return observation, float(reward), terminated, truncated, info

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
