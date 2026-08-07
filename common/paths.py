"""Project paths. Resolves the repo root from FRAUD_RADAR_ROOT if set,
otherwise by walking up to the nearest pyproject.toml.

Set FRAUD_RADAR_ROOT in any environment without a repo checkout (containers).
"""

import os
from pathlib import Path


def _find_root(start: Path, marker: str = "pyproject.toml") -> Path:
    if env := os.environ.get("FRAUD_RADAR_ROOT"):
        return Path(env).resolve()
    for d in [start, *start.parents]:
        if (d / marker).exists():
            return d
    raise FileNotFoundError(
        f"No {marker} above {start}. Set FRAUD_RADAR_ROOT to the project root."
    )


ROOT = _find_root(Path(__file__).resolve())

DATA = ROOT / "data"
ARTIFACTS = ROOT / "model" / "artifacts"
MODEL = ARTIFACTS / "isolation_forest.pkl"
