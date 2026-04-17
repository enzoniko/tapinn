from __future__ import annotations

import json
import importlib.util
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "exp_common" / "experiments"
_experiments_pkg = types.ModuleType("exp_common.experiments")
_experiments_pkg.__path__ = [str(_EXPERIMENTS_DIR)]
sys.modules.setdefault("exp_common.experiments", _experiments_pkg)

_exp4_spec = importlib.util.spec_from_file_location(
    "exp_common.experiments.exp4_sensitivity",
    _EXPERIMENTS_DIR / "exp4_sensitivity.py",
)
assert _exp4_spec is not None and _exp4_spec.loader is not None
_exp4_module = importlib.util.module_from_spec(_exp4_spec)
sys.modules[_exp4_spec.name] = _exp4_module
_exp4_spec.loader.exec_module(_exp4_module)
run_exp_4_sensitivity_and_robustness = _exp4_module.run_exp_4_sensitivity_and_robustness


class Exp4UpdatedTest(unittest.TestCase):
    temp_root: Path = Path(".")

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = Path(tempfile.mkdtemp(prefix="exp4_updated_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    def test_smoke_run_emits_minimum_sensitivity_outputs(self) -> None:
        output_root = self.temp_root / "smoke"
        payload = run_exp_4_sensitivity_and_robustness(
            output_root=str(output_root),
            device_name="cpu",
            smoke_test=True,
            seed=123,
        )

        self.assertEqual(payload["models"], ["StandardPINN", "StandardPINN_OC"])
        self.assertEqual(payload["max_epochs"], 4)
        self.assertFalse(payload["all_configs"])
        self.assertIn("noise_sweep", payload)
        self.assertIn("window_sweep", payload)
        self.assertIn("oc_noise_benefit", payload)

        run_dir = output_root / "exp_4_sensitivity_and_robustness"
        noise_path = run_dir / "tables" / "noise_sweep.csv"
        window_path = run_dir / "tables" / "window_sweep.csv"
        benefit_path = run_dir / "tables" / "oc_noise_benefit.csv"
        results_path = run_dir / "results.json"

        self.assertTrue(noise_path.exists())
        self.assertTrue(window_path.exists())
        self.assertTrue(benefit_path.exists())
        self.assertTrue(results_path.exists())

        noise_text = noise_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN", noise_text)
        self.assertIn("StandardPINN_OC", noise_text)
        self.assertIn("relative_l2_error_mean", noise_text)
        self.assertIn("disambiguation_score_mean", noise_text)

        window_text = window_path.read_text(encoding="utf-8")
        self.assertIn("window_fraction", window_text)
        self.assertIn("forecast_error_mean", window_text)
        self.assertIn("observation_steps", window_text)

        benefit_text = benefit_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN_OC", benefit_text)
        self.assertIn("forecast_error_delta", benefit_text)

        payload_from_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(payload_from_disk["models"], ["StandardPINN", "StandardPINN_OC"])
        self.assertEqual(len(payload_from_disk["noise_sweep"]), 12)
        self.assertEqual(len(payload_from_disk["window_sweep"]), 8)
        self.assertEqual(len(payload_from_disk["oc_noise_benefit"]), 6)


if __name__ == "__main__":
    unittest.main()
