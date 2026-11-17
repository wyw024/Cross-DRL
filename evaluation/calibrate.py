"""Traffic-scale and ego-car smoke checks (no model training).

Usage (from repo root):
    python -m evaluation.calibrate --seeds 10 --routes straight left right
"""
import argparse
import contextlib
import io
import json
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sumolib
from gymnasium.utils import seeding

from environment.sumo_env import (DENSITY_SCALE, EGO_ID, EGO_ROUTES, SUMO_CONFIG,
                                  IntersectionEnv)

ROUTES_FILE = ROOT / 'environment' / 'network' / 'routes.rou.xml'
MAX_STEPS = 300


def env_sumo_seed(seed):
    """Same derivation as IntersectionEnv.reset(): first draw from the seeded np_random."""
    rng, _ = seeding.np_random(seed)
    return int(rng.integers(0, 2**31 - 1))


def expected_inserted(scale):
    """Expected number of background vehicles from routes.rou.xml probability flows."""
    total = 0.0
    for flow in ET.parse(ROUTES_FILE).getroot().findall('flow'):
        total += float(flow.get('probability')) * (float(flow.get('end')) - float(flow.get('begin')))
    return total * scale


def background_run(density, scale, seed):
    """Background-only run with the same SUMO options the environment uses."""
    with tempfile.TemporaryDirectory(prefix='sumo_calibrate_') as tmp:
        summary = Path(tmp) / 'summary.xml'
        cmd = [sumolib.checkBinary('sumo'), '-c', SUMO_CONFIG,
               '--step-length', '1', '--seed', str(env_sumo_seed(seed)), '--scale', str(scale),
               '--collision.action', 'warn', '--time-to-teleport', '300',
               '--no-step-log', 'true', '--no-warnings', 'true', '--duration-log.disable', 'true',
               '--summary-output', str(summary)]
        subprocess.run(cmd, check=True, capture_output=True)
        rows = ET.parse(summary).getroot().findall('step')
    last = rows[-1]
    early = [r for r in rows if float(r.get('time')) <= 60]      # window the ego actually sees
    g = lambda r, k: int(r.get(k))
    return {
        'kind': 'background', 'density': density, 'seed': seed, 'sumo_seed': env_sumo_seed(seed),
        'end_time_s': float(last.get('time')),
        'inserted': g(last, 'inserted'), 'expected_inserted': round(expected_inserted(scale), 1),
        'ended': g(last, 'ended'), 'running_at_end': g(last, 'running'),
        'waiting_at_end': g(last, 'waiting'),                    # insertion backlog
        'peak_waiting': max(g(r, 'waiting') for r in rows),
        'teleports': g(last, 'teleports'), 'collisions': g(last, 'collisions'),
        'peak_halting': max(g(r, 'halting') for r in rows),
        'mean_halting': sum(g(r, 'halting') for r in rows) / len(rows),
        'running_at_60s': g(early[-1], 'running'),
        'mean_halting_first_60s': sum(g(r, 'halting') for r in early) / len(early),
    }


def force_ego_red(env, seconds=40):
    c = env.connection
    next_tls = c.vehicle.getNextTLS(EGO_ID)
    if not next_tls:
        return False
    tls, index, _, _ = next_tls[0]
    phases = c.trafficlight.getAllProgramLogics(tls)[0].phases
    red = next(i for i, p in enumerate(phases) if p.state[index].lower() == 'r')
    c.trafficlight.setPhase(tls, red)
    c.trafficlight.setPhaseDuration(tls, seconds)
    return True


def outcome(info):
    if info.get('collision'):
        return 'collision'
    if info.get('red_light_violation'):
        return 'red_violation'
    if info.get('teleported'):
        return 'teleported'
    if info.get('passed_intersection'):
        return 'passed'
    if info.get('arrived'):
        return 'arrived'
    if not info.get('ego_present', True):
        return 'ego_missing'
    return 'timeout'


def ego_run(density, route, mode, seed):
    env = IntersectionEnv(traffic_density=density, ego_route=route, max_steps=MAX_STEPS)
    try:
        with contextlib.redirect_stdout(io.StringIO()):             # env prints [SUMO INFO]
            _, info0 = env.reset(seed=seed)
            assert info0['sumo_seed'] == env_sumo_seed(seed), 'seed derivation drifted from env'
            forced = force_ego_red(env) if mode == 'controlled_red' else None
            infos, gaps = [], []
            for _ in range(MAX_STEPS):
                _, _, terminated, truncated, info = env.step(4)      # always +2 m/s^2; SUMO still enforces safety
                infos.append(info)
                gaps.append(env.current_leader_gap)                  # not exposed in info
                if terminated or truncated:
                    break
    finally:
        env.close()
    seen = [x for x in gaps if x is not None]
    return {
        'kind': 'ego', 'mode': mode, 'density': density, 'route': route, 'seed': seed,
        'forced_red_ok': forced, 'steps': len(infos), 'outcome': outcome(infos[-1]),
        'zero_speed_steps': sum(x.get('speed_m_s', 1) < 0.1 for x in infos),
        'red_signal_steps': sum(x.get('current_signal') == 'red' for x in infos),
        'leader_observed_steps': len(seen),
        'min_leader_gap_m': min(seen) if seen else None,
        'initial_phase_offset_s': info0['initial_phase_offset_s'],
    }


def summarize(records):
    out = {}
    bg = [r for r in records if r['kind'] == 'background']
    for d in DENSITY_SCALE:
        rs = [r for r in bg if r['density'] == d]
        if not rs:
            continue
        f = lambda k: (round(statistics.mean(r[k] for r in rs), 2),
                       round(statistics.stdev(r[k] for r in rs), 2) if len(rs) > 1 else 0.0)
        out[f'background/{d}'] = {k: f(k) for k in ('inserted', 'expected_inserted', 'running_at_end',
                                                    'peak_waiting', 'peak_halting', 'mean_halting',
                                                    'running_at_60s', 'mean_halting_first_60s')}
    ego = [r for r in records if r['kind'] == 'ego']
    keys = sorted({(r['mode'], r['route'], r['density']) for r in ego})
    for mode, route, d in keys:
        rs = [r for r in ego if (r['mode'], r['route'], r['density']) == (mode, route, d)]
        counts = {}
        for r in rs:
            counts[r['outcome']] = counts.get(r['outcome'], 0) + 1
        out[f'ego/{mode}/{route}/{d}'] = {
            'n': len(rs), 'outcomes': counts,
            'median_steps': statistics.median(r['steps'] for r in rs),
            'mean_zero_speed_steps': round(statistics.mean(r['zero_speed_steps'] for r in rs), 1),
            'mean_leader_observed_steps': round(statistics.mean(r['leader_observed_steps'] for r in rs), 1),
            'min_leader_gap_m': min((r['min_leader_gap_m'] for r in rs if r['min_leader_gap_m'] is not None),
                                    default=None),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=42)
    p.add_argument('--routes', nargs='+', default=['straight'], choices=list(EGO_ROUTES))
    p.add_argument('--modes', nargs='+', default=['natural', 'controlled_red'],
                   choices=['natural', 'controlled_red'])
    p.add_argument('--out', type=Path, default=ROOT / 'results' / 'logs' / 'calibration.json')
    args = p.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.seeds)
    records = []
    for density, scale in DENSITY_SCALE.items():
        for seed in seeds:
            records.append(background_run(density, scale, seed))
            for route in args.routes:
                for mode in args.modes:
                    records.append(ego_run(density, route, mode, seed))
        print(f'[{density}] done', file=sys.stderr)

    result = {'config': {'seeds': list(seeds), 'routes': args.routes, 'modes': args.modes,
                         'max_steps': MAX_STEPS},
              'summary': summarize(records), 'records': records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    print('Saved', args.out)


if __name__ == '__main__':
    main()
