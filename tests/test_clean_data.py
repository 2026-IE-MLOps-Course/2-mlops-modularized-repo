# test_clean_data.py
"""
Educational Goal:
- Verify that the stateless data cleaning rules work as intended.
- Keep tests isolated from file I/O to test only the cleaning logic.
"""

import pandas as pd
import pytest
from src.clean_data import clean_dataframe

TARGET_COLUMN = "OD"


def test_clean_dataframe_happy_path_contract():
    """
    Unit Test: Standardize columns, drop ID, remove duplicates and missing rows, reset index.
    """
    # 1. SETUP: Create deterministic messy data in-memory
    df_messy = pd.DataFrame({
        "ID": [1, 2, 3, 3],
        "rx ds": [10, 20, pd.NA, 10],
        "OD": [0, 1, 0, 0],
    })

    # 2. EXECUTE
    df_clean = clean_dataframe(df_messy, target_column=TARGET_COLUMN)

    # 3. ASSERT
    assert "ID" not in df_clean.columns
    assert "rx_ds" in df_clean.columns
    assert "rx ds" not in df_clean.columns
    assert TARGET_COLUMN in df_clean.columns

    assert len(df_clean) == 2, "Failed to accurately drop NA and duplicate rows"
    assert df_clean.index.equals(pd.RangeIndex(
        len(df_clean))), "Index was not reset"


def test_clean_dataframe_raises_on_none_input():
    """
    Unit Test: None input must fail fast.
    """
    with pytest.raises(ValueError, match="df_raw is None"):
        clean_dataframe(None, target_column=TARGET_COLUMN)


def test_clean_dataframe_raises_if_target_missing():
    """
    Unit Test: Missing target column must fail fast.
    """
    df_no_target = pd.DataFrame({"feature_1": [1, 2]})

    with pytest.raises(ValueError, match="target column"):
        clean_dataframe(df_no_target, target_column=TARGET_COLUMN)
