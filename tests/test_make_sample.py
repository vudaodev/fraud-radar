"""Tests for deterministic test and demo sample generation."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import make_sample


SOURCE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


@pytest.fixture
def sample_environment(tmp_path, monkeypatch):
    """Create a source dataset with enough rows for both configured modes."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_path = data_dir / "creditcard.csv"

    legitimate_rows = 3000
    fraud_rows = 60
    total_rows = legitimate_rows + fraud_rows
    source = pd.DataFrame(
        {
            "Time": [index // 2 for index in range(total_rows)],
            **{
                f"V{i}": [float(index) for index in range(total_rows)]
                for i in range(1, 29)
            },
            "Amount": [float(index) / 10 for index in range(total_rows)],
            "Class": [0] * legitimate_rows + [1] * fraud_rows,
        }
    )[SOURCE_COLUMNS]
    source.to_csv(source_path, index=False)

    monkeypatch.setattr(make_sample, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(make_sample, "DATA_PATH", source_path)
    return source


@pytest.mark.parametrize("mode", ["test", "demo"])
def test_generates_configured_sample(sample_environment, mode):
    output_path = make_sample.make_sample(mode)
    result = pd.read_csv(output_path)
    config = make_sample.CONFIGS[mode]

    assert result.columns.tolist() == sample_environment.columns.tolist()
    assert result["Time"].is_monotonic_increasing
    assert len(result) == config["n_fraud"] + config["n_legit"]
    assert result["Class"].value_counts().to_dict() == {
        0: config["n_legit"],
        1: config["n_fraud"],
    }


def test_output_is_deterministic(sample_environment):
    output_path = make_sample.make_sample("test")
    first_output = output_path.read_bytes()

    make_sample.make_sample("test")

    assert output_path.read_bytes() == first_output


@pytest.mark.parametrize(
    ("source_columns", "source_rows", "expected_error"),
    [
        (["Class"], [{"Class": 0}], "missing required columns: Time"),
        (
            SOURCE_COLUMNS,
            [{column: 0 for column in SOURCE_COLUMNS}],
            "does not contain enough rows",
        ),
    ],
)
def test_invalid_source_does_not_replace_existing_output(
    tmp_path, monkeypatch, source_columns, source_rows, expected_error
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_path = data_dir / "creditcard.csv"
    pd.DataFrame(source_rows, columns=source_columns).to_csv(source_path, index=False)
    output_path = data_dir / "test_sample.csv"
    output_path.write_text("existing output\n")

    monkeypatch.setattr(make_sample, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(make_sample, "DATA_PATH", source_path)

    with pytest.raises(ValueError, match=expected_error):
        make_sample.make_sample("test")

    assert output_path.read_text() == "existing output\n"


def test_missing_source_does_not_replace_existing_output(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_path = data_dir / "test_sample.csv"
    output_path.write_text("existing output\n")

    monkeypatch.setattr(make_sample, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(make_sample, "DATA_PATH", data_dir / "creditcard.csv")

    with pytest.raises(FileNotFoundError, match="Source data not found"):
        make_sample.make_sample("test")

    assert output_path.read_text() == "existing output\n"


def test_rejects_unknown_mode(sample_environment):
    with pytest.raises(ValueError, match="Unknown mode"):
        make_sample.make_sample("unknown")
