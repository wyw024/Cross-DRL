"""Unit tests that need neither SUMO nor PyTorch: CSV merging, seeds, epsilon schedule, aggregation."""
import csv
import math
import tempfile
import unittest
from pathlib import Path

from evaluation.common import SUMMARY_FIELDS, merge_csv, save_csv, densities_from_arg
from evaluation.paths import discover_train_seeds, result_dir, run_dirs
from training.common import (ScenarioSampler, auto_epsilon_decay, env_seed, moving_average,
                             resolve_epsilon_decay)


def summary_row(algo, density, route, reward):
    row = {k: "" for k in SUMMARY_FIELDS}
    row.update(algorithm=algo, density=density, ego_route=route, episodes=3, mean_reward=reward)
    return row


class MergeCsvTest(unittest.TestCase):
    def test_other_routes_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation_summary.csv"
            save_csv([summary_row("dqn", "low", "straight", 70)], SUMMARY_FIELDS, path)
            kept = merge_csv([summary_row("dqn", "low", "left", 69)], SUMMARY_FIELDS, path)
            self.assertEqual(kept, 1)
            with path.open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([(r["density"], r["ego_route"]) for r in rows],
                             [("low", "straight"), ("low", "left")])

    def test_same_key_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            save_csv([summary_row("dqn", "low", "left", 1)], SUMMARY_FIELDS, path)
            merge_csv([summary_row("dqn", "low", "left", 2)], SUMMARY_FIELDS, path)
            with path.open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["mean_reward"], "2")

    def test_density_arg(self):
        self.assertEqual(densities_from_arg("all"), ("low", "medium", "high"))
        self.assertEqual(densities_from_arg("all+extreme")[-1], "extreme")
        self.assertEqual(densities_from_arg("high"), ("high",))


class TrainingHelpersTest(unittest.TestCase):
    def test_epsilon_reaches_floor(self):
        decay = auto_epsilon_decay(1.0, 0.05, 600, 0.7)
        self.assertAlmostEqual(decay ** 420, 0.05, places=6)
        self.assertEqual(resolve_epsilon_decay("0.995", 1.0, 0.05, 300, 0.7), 0.995)
        with self.assertRaises(ValueError):
            resolve_epsilon_decay("1.5", 1.0, 0.05, 300, 0.7)

    def test_train_seeds_do_not_hit_test_seeds(self):
        used = {env_seed(s, e) for s in (42, 43, 44) for e in range(1, 2001)}
        self.assertTrue(used.isdisjoint(range(1000, 1500)))

    def test_sampler_is_reproducible_and_covers_all(self):
        a = ScenarioSampler("mixed", ["straight", "left", "right"], 42)
        b = ScenarioSampler("mixed", ["straight", "left", "right"], 42)
        sa = [a.sample() for _ in range(200)]
        self.assertEqual(sa, [b.sample() for _ in range(200)])
        self.assertEqual({d for d, _ in sa}, {"low", "medium", "high"})
        self.assertEqual({r for _, r in sa}, {"straight", "left", "right"})
        single = ScenarioSampler("high", ["left"], 1)
        self.assertEqual({single.sample() for _ in range(10)}, {("high", "left")})

    def test_moving_average(self):
        out = moving_average([1, 2, 3, 4], 2)
        self.assertTrue(all(math.isclose(x, y) for x, y in zip(out, [1, 1.5, 2.5, 3.5])))


class PathsTest(unittest.TestCase):
    def test_seed_discovery_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (43, 42):
                folder = result_dir(root, "dqn", seed)
                folder.mkdir(parents=True)
                (folder / "evaluation_summary.csv").write_text("a\n", encoding="utf-8")
            self.assertEqual(discover_train_seeds(root, "dqn"), [42, 43])
            self.assertEqual([s for s, _ in run_dirs(root, "dqn")], [42, 43])
            self.assertEqual([s for s, _ in run_dirs(root, "dqn", 43)], [43])
            legacy = root / "q_learning"
            legacy.mkdir()
            (legacy / "evaluation_summary.csv").write_text("a\n", encoding="utf-8")
            self.assertEqual(run_dirs(root, "q_learning"), [(None, legacy)])
            self.assertEqual(result_dir(root, "rule_based", 42), root / "rule_based")


if __name__ == "__main__":
    unittest.main()
