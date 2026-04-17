from __future__ import annotations

import json
import importlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

run_exp_3_sota_baselines_and_capacity = importlib.import_module(
    "exp_common.experiments.exp3_capacity"
).run_exp_3_sota_baselines_and_capacity


class Exp3UpdatedTest(unittest.TestCase):
    temp_root: Path = Path(".")

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = Path(tempfile.mkdtemp(prefix="exp3_updated_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    def test_smoke_run_emits_minimum_oc_comparison_outputs(self) -> None:
        output_root = self.temp_root / "smoke"
        payload = run_exp_3_sota_baselines_and_capacity(
            output_root=str(output_root),
            device_name="cpu",
            smoke_test=True,
            seed=123,
        )

        self.assertEqual(payload["models"], ["StandardPINN", "StandardPINN_OC"])
        self.assertEqual(payload["max_epochs"], 4)
        self.assertFalse(payload["all_configs"])

        run_dir = output_root / "exp_3_sota_baselines_and_capacity"
        summary_path = run_dir / "tables" / "model_summary.csv"
        capacity_path = run_dir / "tables" / "capacity_benchmark.csv"
        benefit_path = run_dir / "tables" / "oc_benefit_summary.csv"
        results_path = run_dir / "results.json"

        self.assertTrue(summary_path.exists())
        self.assertTrue(capacity_path.exists())
        self.assertTrue(benefit_path.exists())
        self.assertTrue(results_path.exists())

        summary_text = summary_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN", summary_text)
        self.assertIn("StandardPINN_OC", summary_text)
        self.assertIn("param_count", summary_text)
        self.assertIn("generalization_gap_mean", summary_text)

        benefit_text = benefit_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN_OC", benefit_text)
        self.assertIn("param_count_delta", benefit_text)

        payload_from_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(payload_from_disk["models"], ["StandardPINN", "StandardPINN_OC"])
        self.assertEqual(len(payload_from_disk["summary"]), 4)
        self.assertEqual(len(payload_from_disk["oc_benefit"]), 2)
        self.assertEqual(len(payload_from_disk["capacity_benchmark"]), 2)


if __name__ == "__main__":
    unittest.main()
