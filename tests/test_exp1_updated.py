from __future__ import annotations

import json
import importlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

run_exp_1_ode_chaos_suite = importlib.import_module("exp_common.experiments.exp1_ode_chaos").run_exp_1_ode_chaos_suite


class Exp1UpdatedTest(unittest.TestCase):
    temp_root: Path = Path(".")

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = Path(tempfile.mkdtemp(prefix="exp1_updated_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    def test_smoke_run_emits_minimum_oc_comparison_outputs(self) -> None:
        output_root = self.temp_root / "smoke"
        payload = run_exp_1_ode_chaos_suite(
            output_root=str(output_root),
            device_name="cpu",
            smoke_test=True,
            seed=123,
        )

        self.assertEqual(payload["models"], ["StandardPINN", "StandardPINN_OC"])
        self.assertEqual(payload["max_epochs"], 4)
        self.assertFalse(payload["all_configs"])

        run_dir = output_root / "exp_1_ode_chaos_suite"
        summary_path = run_dir / "tables" / "model_summary.csv"
        benefit_path = run_dir / "tables" / "oc_benefit_summary.csv"
        results_path = run_dir / "results.json"

        self.assertTrue(summary_path.exists())
        self.assertTrue(benefit_path.exists())
        self.assertTrue(results_path.exists())

        summary_text = summary_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN", summary_text)
        self.assertIn("StandardPINN_OC", summary_text)
        self.assertIn("relative_l2_error_mean", summary_text)
        self.assertIn("disambiguation_score_mean", summary_text)

        benefit_text = benefit_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN_OC", benefit_text)

        payload_from_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(payload_from_disk["models"], ["StandardPINN", "StandardPINN_OC"])
        self.assertEqual(len(payload_from_disk["summary"]), 6)
        self.assertEqual(len(payload_from_disk["oc_benefit"]), 3)


if __name__ == "__main__":
    unittest.main()
