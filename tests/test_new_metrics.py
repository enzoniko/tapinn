from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_common.metrics import compute_disambiguation_score, compute_relative_l2_error


def test_rel_l2_correct_values() -> None:
    pred = np.array([1.1, 2.2, 3.3])
    target = np.array([1.0, 2.0, 3.0])
    expected = np.linalg.norm(pred - target) / np.linalg.norm(target)
    assert abs(compute_relative_l2_error(pred, target) - expected) < 1e-10


def test_rel_l2_near_zero_target() -> None:
    pred = np.array([1.0, -2.0])
    target = np.array([0.0, 0.0])
    result = compute_relative_l2_error(pred, target)
    assert np.isfinite(result)
    assert result == np.linalg.norm(pred - target)


def test_disambiguation_perfect_separation() -> None:
    rng = np.random.default_rng(42)
    embeddings = np.vstack(
        [
            rng.normal(loc=[0, 0], scale=0.1, size=(20, 2)),
            rng.normal(loc=[10, 0], scale=0.1, size=(20, 2)),
            rng.normal(loc=[0, 10], scale=0.1, size=(20, 2)),
        ]
    )
    trajectory_ids = [0] * 20 + [1] * 20 + [2] * 20
    assert compute_disambiguation_score(embeddings, trajectory_ids) > 0.9


def test_disambiguation_overlap() -> None:
    embeddings = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    trajectory_ids = [0, 1, 2, 3, 0, 1, 2, 3]
    assert compute_disambiguation_score(embeddings, trajectory_ids) < 0.3


def test_disambiguation_single_id() -> None:
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
    trajectory_ids = [3, 3]
    assert compute_disambiguation_score(embeddings, trajectory_ids) == 0.0
