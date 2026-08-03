"""Generate deterministic fraud-detection sample datasets from the source CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "creditcard.csv"

CONFIGS = {
    "test": {"n_fraud": 20, "n_legit": 80, "out": "data/test_sample.csv"},
    "demo": {"n_fraud": 50, "n_legit": 2950, "out": "data/demo_sample.csv"},
}

REQUIRED_COLUMNS = {"Time", "Class"}
RANDOM_STATE = 42


def make_sample(mode: str) -> Path:
    """Create the configured deterministic sample and return its output path."""
    try:
        config = CONFIGS[mode]
    except KeyError:
        valid_modes = ", ".join(CONFIGS)
        raise ValueError(f"Unknown mode {mode!r}. Expected one of: {valid_modes}.") from None

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Source data not found: {DATA_PATH}")

    source = pd.read_csv(DATA_PATH)
    missing_columns = REQUIRED_COLUMNS.difference(source.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Source data is missing required columns: {missing}")

    fraud = source.loc[source["Class"] == 1]
    legitimate = source.loc[source["Class"] == 0]
    n_fraud = config["n_fraud"]
    n_legit = config["n_legit"]

    if len(fraud) < n_fraud or len(legitimate) < n_legit:
        raise ValueError(
            "Source data does not contain enough rows for "
            f"{mode!r}: requested {n_fraud} fraud and {n_legit} legitimate rows; "
            f"found {len(fraud)} fraud and {len(legitimate)} legitimate rows."
        )

    sample = pd.concat(
        [
            fraud.sample(n=n_fraud, replace=False, random_state=RANDOM_STATE),
            legitimate.sample(n=n_legit, replace=False, random_state=RANDOM_STATE),
        ]
    ).sort_values("Time", kind="mergesort")

    output_path = REPO_ROOT / config["out"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)

    print(
        f"Generated {mode} sample at {output_path}: "
        f"{len(sample)} rows ({n_fraud} fraud, {n_legit} legitimate)."
    )
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sample generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=CONFIGS, required=True, help="Sample profile to generate.")
    return parser.parse_args()


def main() -> None:
    """Generate the requested sample profile."""
    make_sample(parse_args().mode)


if __name__ == "__main__":
    main()
