# tests/test_features.py

"""
Educational Goal:
- Why this test exists in an MLOps system: Validate the feature engineering contract without relying on training code.
- Responsibility (separation of concerns): Ensure get_feature_preprocessor builds a valid ColumnTransformer and fails fast on bad config.
- Pipeline contract (inputs and outputs): Returns a configured ColumnTransformer or raises early when config is invalid.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features import get_feature_preprocessor


def _transformer_names(preprocessor: ColumnTransformer) -> list:
    """Helper to extract step names from a ColumnTransformer."""
    return [name for name, _, _ in preprocessor.transformers]


# --------------------------------------------------------
# 1) FAIL FAST: Empty configuration raises
# --------------------------------------------------------
def test_get_feature_preprocessor_raises_on_empty_config():
    """
    Inputs: All feature lists empty
    Outputs: ValueError
    Why this contract matters for reliable ML delivery:
    - Empty recipes should crash at build time, not halfway through training.
    """
    with pytest.raises(ValueError, match=r"No feature columns configured"):
        get_feature_preprocessor(
            quantile_bin_cols=[],
            categorical_onehot_cols=[],
            numeric_passthrough_cols=[],
            binary_sum_cols=[],
            n_bins=4,
        )

# --------------------------------------------------------
# 2) FAIL FAST: Invalid bin count raises
# --------------------------------------------------------


def test_get_feature_preprocessor_raises_on_invalid_n_bins():
    """
    Inputs: n_bins = 1
    Outputs: ValueError
    Why this contract matters for reliable ML delivery:
    - KBinsDiscretizer requires at least 2 bins. We fail fast before Scikit-Learn throws a cryptic error.
    """
    with pytest.raises(ValueError, match=r"n_bins must be >= 2"):
        get_feature_preprocessor(
            quantile_bin_cols=["rx_ds"],
            n_bins=1
        )


# --------------------------------------------------------
# 3) CONTRACT: Returns ColumnTransformer and registers expected blocks
# --------------------------------------------------------
def test_get_feature_preprocessor_returns_columntransformer_and_registers_blocks():
    """
    Inputs: Valid configuration with multiple transformer types
    Outputs: ColumnTransformer with expected named blocks
    Why this contract matters for reliable ML delivery:
    - A stable recipe boundary lets train.py stay simple and consistent.
    """
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["rx_ds"],
        categorical_onehot_cols=["gender"],
        numeric_passthrough_cols=["age"],
        binary_sum_cols=["P_D", "P_P"],
        n_bins=4,
    )

    assert isinstance(preprocessor, ColumnTransformer)

    names = _transformer_names(preprocessor)

    # We check for inclusion rather than exact length so the test isn't brittle
    assert "quantile_bins" in names
    assert "cat_ohe" in names
    assert "binary_sum" in names
    assert "num_pass" in names


# --------------------------------------------------------
# 4) CORRECTNESS: binary_sum computes row-wise sums when executed
# --------------------------------------------------------
def test_binary_sum_logic_executes_correctly():
    """
    Inputs: A recipe containing only the binary_sum transformer + dummy DataFrame
    Outputs: Transformed output equals expected row-wise sums
    Why this contract matters for reliable ML delivery:
    - Derived features must be deterministic and consistent across train and inference.
    """
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=[],
        categorical_onehot_cols=[],
        numeric_passthrough_cols=[],
        binary_sum_cols=["P_D", "P_P", "S_P"],
        n_bins=4,
    )

    df_dummy = pd.DataFrame(
        {
            "P_D": [1, 0, 1],
            "P_P": [1, 0, 0],
            "S_P": [0, 0, 1],
        }
    )

    transformed = preprocessor.fit_transform(df_dummy)

    # Expected sums: Row 0: 2, Row 1: 0, Row 2: 2
    expected = np.array([[2], [0], [2]])

    transformed_values = np.asarray(transformed)

    assert transformed_values.shape == expected.shape
    # assert_allclose is tolerant to Scikit-Learn silently casting to float64
    np.testing.assert_allclose(transformed_values, expected, atol=1e-5)
