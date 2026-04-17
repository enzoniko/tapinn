from __future__ import annotations
# pyright: reportAny=false, reportUnusedCallResult=false, reportImplicitOverride=false

import json
import importlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

run_exp_5_theoretical_optimization_landscape = importlib.import_module(
    "exp_common.experiments.exp5_ntk_landscape"
).run_exp_5_theoretical_optimization_landscape


class Exp5UpdatedTest(unittest.TestCase):
    temp_root: Path = Path(".")

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = Path(tempfile.mkdtemp(prefix="exp5_updated_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    def test_smoke_run_emits_soft_ao_comparison_outputs(self) -> None:
        output_root = self.temp_root / "smoke"
        payload = run_exp_5_theoretical_optimization_landscape(
            output_root=str(output_root),
            device_name="cpu",
            smoke_test=True,
            seed=123,
        )

        self.assertEqual(payload["models"], ["StandardPINN", "StandardPINN_OC"])
        self.assertEqual(payload["problems"], ["duffing"])
        self.assertEqual(payload["max_epochs"], 20)
        self.assertEqual(payload["checkpoint_step"], 10)
        self.assertFalse(payload["all_configs"])

        run_dir = output_root / "exp_5_theoretical_optimization_landscape"
        seed_summary_path = run_dir / "tables" / "seed_summary.csv"
        model_summary_path = run_dir / "tables" / "model_summary.csv"
        soft_ao_path = run_dir / "tables" / "soft_ao_summary.csv"
        results_path = run_dir / "results.json"

        self.assertTrue(seed_summary_path.exists())
        self.assertTrue(model_summary_path.exists())
        self.assertTrue(soft_ao_path.exists())
        self.assertTrue(results_path.exists())

        seed_text = seed_summary_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN", seed_text)
        self.assertIn("StandardPINN_OC_ao", seed_text)
        self.assertIn("StandardPINN_OC_joint", seed_text)
        self.assertIn("relative_l2_error", seed_text)
        self.assertIn("oc_variant", seed_text)

        model_summary_text = model_summary_path.read_text(encoding="utf-8")
        self.assertIn("final_condition_number_mean", model_summary_text)
        self.assertIn("training_mode", model_summary_text)

        soft_ao_text = soft_ao_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN_OC", soft_ao_text)
        self.assertIn("condition_number_delta", soft_ao_text)

        payload_from_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(payload_from_disk["problems"], ["duffing"])
        self.assertEqual(len(payload_from_disk["seed_summary"]), 3)
        self.assertEqual(len(payload_from_disk["soft_ao_summary"]), 1)


if __name__ == "__main__":
    unittest.main()
