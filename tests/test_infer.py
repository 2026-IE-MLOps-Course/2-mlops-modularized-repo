# tests/test_infer.py
"""
Educational Goal:
- Why this test exists in an MLOps system: Ensure the inference module strictly honors the pipeline contract.
- Responsibility: Verify that predictions are formatted correctly (preserving indexes), probabilities are mathematically sound, and bad inputs crash early.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.infer import run_inference


@pytest.fixture
def dummy_inference_data():
    """Provides a minimal feature DataFrame with a custom index to test index preservation."""
    X = pd.DataFrame(
        {"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]},
        index=["patient_10", "patient_11", "patient_12", "patient_13"]
    )
    y = pd.Series([0, 1, 0, 1], index=X.index)
    return X, y


@pytest.fixture
def dummy_classifier(dummy_inference_data):
    """Mock classification model supporting predict() and predict_proba()."""
    X, y = dummy_inference_data
    model = DummyClassifier(strategy="prior")
    model.fit(X, y)
    return model


@pytest.fixture
def dummy_regressor(dummy_inference_data):
    """Mock regression model supporting predict() but NOT predict_proba()."""
    X, y = dummy_inference_data
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    return model


# --------------------------------------------------------
# 1) HAPPY PATH: Output Schema & Index Preservation
# --------------------------------------------------------
def test_inference_returns_dataframe_and_preserves_index(dummy_classifier, dummy_inference_data):
    """Ensure standard inference returns the exact correct shape and retains the patient IDs."""
    X, _ = dummy_inference_data
    df_pred = run_inference(model=dummy_classifier,
                            X_infer=X, include_proba=False)

    assert isinstance(df_pred, pd.DataFrame)
    assert list(df_pred.columns) == ["prediction"]
    assert len(df_pred) == len(X)

    # Critical MLOps check: the original index must be perfectly preserved
    pd.testing.assert_index_equal(df_pred.index, X.index)


def test_inference_include_proba_adds_valid_probability(dummy_classifier, dummy_inference_data):
    """Ensure probabilities are extracted correctly and are mathematically valid."""
    X, _ = dummy_inference_data
    df_pred = run_inference(model=dummy_classifier,
                            X_infer=X, include_proba=True)

    assert "prediction" in df_pred.columns
    assert "proba" in df_pred.columns

    # Mathematical guardrails: Must be float and bounded between 0.0 and 1.0
    assert pd.api.types.is_float_dtype(df_pred["proba"])
    assert (df_pred["proba"] >= 0.0).all()
    assert (df_pred["proba"] <= 1.0).all()


# --------------------------------------------------------
# 2) CONTRACT CHECKS: Model Guardrails
# --------------------------------------------------------
def test_raises_if_model_has_no_predict(dummy_inference_data):
    """Crash immediately if the loaded artifact is not a Scikit-Learn estimator."""
    X, _ = dummy_inference_data

    class BadArtifact:
        pass

    with pytest.raises(TypeError, match="Fatal: model must implement predict"):
        run_inference(model=BadArtifact(), X_infer=X, include_proba=False)


def test_raises_if_proba_requested_but_unsupported(dummy_regressor, dummy_inference_data):
    """Crash if a user asks for probabilities from a model that cannot provide them."""
    X, _ = dummy_inference_data

    with pytest.raises(TypeError, match="Fatal: include_proba=True but the loaded model does not support predict_proba"):
        run_inference(model=dummy_regressor, X_infer=X, include_proba=True)


# --------------------------------------------------------
# 3) FAIL FAST: Structural Guardrails
# --------------------------------------------------------
def test_raises_on_empty_X_infer(dummy_classifier, dummy_inference_data):
    """Crash if an empty payload is sent to the inference engine."""
    X, _ = dummy_inference_data
    X_empty = X.iloc[0:0]

    with pytest.raises(ValueError, match="Fatal: X_infer is empty"):
        run_inference(model=dummy_classifier,
                      X_infer=X_empty, include_proba=False)


def test_raises_if_X_infer_not_dataframe(dummy_classifier, dummy_inference_data):
    """Crash if X_infer is a raw numpy array, preventing silent feature-name drops."""
    X, _ = dummy_inference_data
    X_numpy = X.values

    with pytest.raises(TypeError, match="Fatal: X_infer must be a pandas DataFrame"):
        run_inference(model=dummy_classifier,
                      X_infer=X_numpy, include_proba=False)
