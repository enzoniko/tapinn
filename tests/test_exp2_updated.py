from __future__ import annotations
# pyright: reportAny=false, reportImplicitOverride=false, reportUnusedCallResult=false

import json
import importlib.util
import types
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "exp_common" / "experiments"
_experiments_pkg = types.ModuleType("exp_common.experiments")
_experiments_pkg.__path__ = [str(_EXPERIMENTS_DIR)]
sys.modules.setdefault("exp_common.experiments", _experiments_pkg)

_exp2_spec = importlib.util.spec_from_file_location(
    "exp_common.experiments.exp2_pde_well",
    _EXPERIMENTS_DIR / "exp2_pde_well.py",
)
assert _exp2_spec is not None and _exp2_spec.loader is not None
_exp2_module = importlib.util.module_from_spec(_exp2_spec)
sys.modules[_exp2_spec.name] = _exp2_module
_exp2_spec.loader.exec_module(_exp2_module)
run_exp_2_pde_spatiotemporal_suite = _exp2_module.run_exp_2_pde_spatiotemporal_suite


class Exp2UpdatedTest(unittest.TestCase):
    temp_root: Path = Path(".")

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = Path(tempfile.mkdtemp(prefix="exp2_updated_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    def test_smoke_run_emits_minimum_oc_comparison_outputs(self) -> None:
        output_root = self.temp_root / "smoke"
        payload = run_exp_2_pde_spatiotemporal_suite(
            output_root=str(output_root),
            device_name="cpu",
            smoke_test=True,
            seed=123,
        )

        self.assertEqual(payload["models"], ["StandardPINN", "StandardPINN_OC"])
        self.assertEqual(payload["max_epochs"], 4)
        self.assertFalse(payload["all_configs"])

        run_dir = output_root / "exp_2_pde_spatiotemporal_suite"
        seed_metrics_path = run_dir / "tables" / "seed_metrics.csv"
        summary_path = run_dir / "tables" / "model_summary.csv"
        benefit_path = run_dir / "tables" / "oc_benefit_summary.csv"
        legacy_path = run_dir / "tables" / "summary.csv"
        results_path = run_dir / "results.json"

        self.assertTrue(seed_metrics_path.exists())
        self.assertTrue(summary_path.exists())
        self.assertTrue(benefit_path.exists())
        self.assertTrue(legacy_path.exists())
        self.assertTrue(results_path.exists())

        seed_metrics_text = seed_metrics_path.read_text(encoding="utf-8")
        self.assertIn("family", seed_metrics_text)
        self.assertIn("with_oc", seed_metrics_text)
        self.assertIn("oc_variant", seed_metrics_text)
        self.assertIn("relative_l2_error", seed_metrics_text)
        self.assertIn("disambiguation_score", seed_metrics_text)

        summary_text = summary_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN", summary_text)
        self.assertIn("StandardPINN_OC", summary_text)
        self.assertIn("relative_l2_error_mean", summary_text)
        self.assertIn("disambiguation_score_mean", summary_text)

        benefit_text = benefit_path.read_text(encoding="utf-8")
        self.assertIn("StandardPINN_OC", benefit_text)
        self.assertIn("relative_l2_error_delta", benefit_text)

        payload_from_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(payload_from_disk["models"], ["StandardPINN", "StandardPINN_OC"])
        # Expected: 3 PDEs × 2 models = 6, + 1 Well × 2 models (shear_flow in smoke) = 8 total
        self.assertEqual(len(payload_from_disk["summary"]), 8)
        # OC benefit: 3 PDEs + 1 Well = 4
        self.assertEqual(len(payload_from_disk["oc_benefit"]), 4)


if __name__ == "__main__":
    unittest.main()
