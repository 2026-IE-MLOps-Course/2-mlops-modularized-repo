# test_load_data.py
"""
Educational Goal:
- Verify the boundary guardrails of the data ingestion step.
- Ensure the pipeline fails fast if files are missing, empty, or misconfigured.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.load_data import load_raw_data

# Constants to prevent copy-paste drift
REAL_DATA_PATH = Path("data/raw/opiod_raw_data.csv")
TARGET_COLUMN = "OD"


def test_load_raw_data_real_dataset_integration():
    """
    Integration Test: Verify the real dataset loads correctly.
    Skips if the dataset is not present so the test suite remains portable.
    """
    if not REAL_DATA_PATH.exists():
        pytest.skip(f"Real dataset not found at {REAL_DATA_PATH}")

    df = load_raw_data(REAL_DATA_PATH)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Real dataset loaded but is empty"
    assert TARGET_COLUMN in df.columns, "Target column missing from real dataset"
    assert len(
        df.columns) >= 3, "Dataset has unexpectedly few columns, check the raw file"


def test_load_raw_data_raises_file_not_found(tmp_path: Path):
    """
    Unit Test: Missing file must fail fast with FileNotFoundError.
    """
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_raw_data(missing_path)


def test_load_raw_data_raises_if_path_is_directory(tmp_path: Path):
    """
    Unit Test: Pointing to a folder instead of a file must fail fast.
    """
    dir_path = tmp_path / "raw_dir"
    dir_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="directory, not a file"):
        load_raw_data(dir_path)


def test_load_raw_data_raises_on_empty_csv(tmp_path: Path):
    """
    Unit Test: An empty CSV must fail fast.
    Covers both a header-only CSV and a completely blank (0 byte) file.
    """
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("col1,col2\n")

    truly_empty = tmp_path / "truly_empty.csv"
    truly_empty.write_text("")

    # Pandas raises different internal errors for these two cases,
    # but our pipeline correctly shapes both into ValueErrors.
    with pytest.raises(ValueError):
        load_raw_data(header_only)

    with pytest.raises(ValueError):
        load_raw_data(truly_empty)
