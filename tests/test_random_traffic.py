"""Check seeded traffic generation and the environment's seed lifecycle."""
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

from environment.sumo_env import IntersectionEnv, SUMO_CONFIG
from sumolib import checkBinary


def traffic_snapshot(seed, scale=1.0):
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / 'routes.xml'
        cmd = [checkBinary('sumo'), '-c', SUMO_CONFIG,
               '--seed', str(seed), '--scale', str(scale), '--no-step-log', 'true',
               '--vehroute-output', str(output), '--vehroute-output.exit-times', 'true']
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        root = ET.parse(output).getroot()
        return sorted((v.get('id'), v.get('depart'), v.find('route').get('edges'))
                      for v in root.findall('vehicle'))


class RandomTrafficTest(unittest.TestCase):
    def test_seed_and_scale(self):
        a = traffic_snapshot(42)
        self.assertEqual(a, traffic_snapshot(42))
        self.assertNotEqual(a, traffic_snapshot(43))
        low, high = traffic_snapshot(42, .5), traffic_snapshot(42, 1.5)
        self.assertLess(len(low), len(a))
        self.assertGreater(len(high), len(a))
        destinations = {route.split()[-1] for vehicle, depart, route in a if vehicle.startswith('north.')}
        self.assertEqual(destinations, {'south_out', 'east_out', 'west_out'})
        print('background trips low/medium/high:', len(low), len(a), len(high))

    def test_environment_seed(self):
        env = IntersectionEnv(traffic_density='low')
        try:
            _, first = env.reset(seed=42)
            _, continued = env.reset()
            _, repeated = env.reset(seed=42)
            self.assertNotEqual(first['sumo_seed'], continued['sumo_seed'])
            self.assertEqual(first['sumo_seed'], repeated['sumo_seed'])
            self.assertEqual(first['initial_phase_offset_s'], repeated['initial_phase_offset_s'])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
