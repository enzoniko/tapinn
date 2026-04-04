from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def prepare_run_dir(output_root: str | Path, script_name: str) -> Path:
    root = ensure_dir(output_root)
    run_dir = ensure_dir(root / script_name)
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "tables")
    return run_dir


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2)
