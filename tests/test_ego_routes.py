"""Smoke tests against real SUMO: preselected north-entry routes."""
import unittest
from environment.sumo_env import IntersectionEnv


class EgoRouteTest(unittest.TestCase):
    def test_routes(self):
        for route, exit_edge in [('straight', 'south_out'), ('left', 'east_out'), ('right', 'west_out')]:
            with self.subTest(route=route):
                env = IntersectionEnv(traffic_density='low', ego_route=route, randomize_signal_phase=False)
                try:
                    obs, info = env.reset(seed=42)
                    self.assertEqual(obs.shape, (7,))
                    self.assertEqual(env.action_space.n, 5)
                    self.assertEqual((info['ego_route'], info['exit_edge']), (route, exit_edge))
                    saw_crossing = False
                    for _ in range(300):
                        obs, _, terminated, truncated, info = env.step(4)
                        self.assertTrue(env.observation_space.contains(obs))
                        saw_crossing |= info['crossed_stop_line']
                        if terminated or truncated:
                            break
                    self.assertTrue(saw_crossing)
                    self.assertTrue(terminated and not truncated and info['passed_intersection'])
                    self.assertEqual(info['exit_edge'], exit_edge)
                    self.assertFalse(info['collision'] or info['teleported'])
                finally:
                    env.close()

    def test_invalid_route(self):
        env = IntersectionEnv()
        try:
            with self.assertRaises(ValueError):
                env.reset(options={'ego_route': 'invalid'})
        finally:
            env.close()

    def test_reset_options_switch_routes(self):
        env = IntersectionEnv(traffic_density='low', randomize_signal_phase=False)
        try:
            for route, expected in [('left', 'east_out'), ('right', 'west_out'), ('straight', 'south_out')]:
                obs, info = env.reset(seed=42, options={'ego_route': route})
                self.assertEqual((info['ego_route'], info['exit_edge']), (route, expected))
                self.assertEqual(env.connection.vehicle.getRoute('ego'), ('north_in', expected))
                self.assertTrue(env.observation_space.contains(obs))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
