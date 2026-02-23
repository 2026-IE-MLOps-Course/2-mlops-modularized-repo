# tests/test_validate.py

"""
Educational Goal:
- Why this test exists in an MLOps system: Ensure our "security gate" triggers correctly when bad data arrives.
- Responsibility (separation of concerns): Test validation logic independently of cleaning, features, or training.
- Pipeline contract: validate_dataframe raises ValueError on empty data, missing columns, or constraint violations, and returns True for clean data.
"""

import pandas as pd
import pytest

from src.validate import validate_dataframe


# --------------------------------------------------------
# 1) FAIL FAST: Empty DataFrame
# --------------------------------------------------------
def test_validate_fails_on_empty_dataframe():
    """An empty dataframe should immediately crash the pipeline."""
    df_empty = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(
            df=df_empty,
            required_columns=["A"]
        )


# --------------------------------------------------------
# 2) CONTRACT CHECK: Missing required columns
# --------------------------------------------------------
def test_validate_fails_on_missing_columns():
    """Missing inputs break the downstream recipe, so we catch it early."""
    df_missing_col = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(
            df=df_missing_col,
            required_columns=["A", "B"]
        )


# --------------------------------------------------------
# 3) DOMAIN CONSTRAINT: Invalid target classes
# --------------------------------------------------------
def test_validate_fails_on_invalid_target_values():
    """Classification models break if unexpected target classes appear."""
    df_bad_target = pd.DataFrame({
        "target": [0, 2, 1],  # 2 is an invalid class
        "feature": [10, 20, 30],
    })

    with pytest.raises(ValueError, match="invalid"):
        validate_dataframe(
            df=df_bad_target,
            required_columns=["target", "feature"],
            target_column="target",
            target_allowed_values=[0, 1],
        )


# --------------------------------------------------------
# 4) DOMAIN CONSTRAINT: Negative numeric values
# --------------------------------------------------------
def test_validate_fails_on_negative_values():
    """Domain logic like 'no negative values' must be strictly enforced."""
    df_bad_math = pd.DataFrame({
        "target": [0, 1, 0],
        "days_supply": [10, -5, 30],  # -5 is an invalid negative value
    })

    with pytest.raises(ValueError, match="negative values"):
        validate_dataframe(
            df=df_bad_math,
            required_columns=["target", "days_supply"],
            numeric_non_negative_cols=["days_supply"],
        )


# --------------------------------------------------------
# 5) HAPPY PATH: Valid simple dataset
# --------------------------------------------------------
def test_validate_passes_on_valid_dataframe():
    """Clean data silently passes the gate returning True."""
    df_clean = pd.DataFrame({
        "target": [0, 1, 0],
        "days_supply": [10, 20, 30],
    })

    result = validate_dataframe(
        df=df_clean,
        required_columns=["target", "days_supply"],
        target_column="target",
        target_allowed_values=[0, 1],
        numeric_non_negative_cols=["days_supply"],
    )

    assert result is True
