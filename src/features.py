# src/features.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Isolate feature engineering rules from their execution to prevent leakage and drift.
- Responsibility (separation of concerns): Define the preprocessing recipe as a ColumnTransformer. No file I/O, no .fit() calls here.
- Pipeline contract: Inputs are configuration lists. Output is a scikit-learn ColumnTransformer.

Where do transformations belong?
- Stateful transformations (MUST be fitted on X_train only) belong here. Example: Quantile bin edges.
- Stateless transformations (Math operations) CAN belong here if they are part of the model's unique contract. Example: binary_sum. Putting it here ensures the deployed model calculates it automatically.
- Only put stateless transforms in `clean_data.py` if they are part of a canonical, company-wide data schema used beyond just this model.

Why this prevents leakage:
- The recipe is fitted ONLY inside `pipeline.fit(X_train, y_train)` inside train.py.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import numpy as np
from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, FunctionTransformer

# We avoid using pandas in this module to prevent accidental data manipulation.


def _row_sum_numpy(X) -> np.ndarray:
    """
    Inputs:
    - X: 2D array-like (DataFrame or numpy array) containing binary or numeric indicators.
    Outputs:
    - sums: 2D numpy array of shape (n_rows, 1) containing row-wise sums.

    MLOps Note: We use np.asarray(X) here to ensure that if a Pandas DataFrame 
    is passed, it is converted to a NumPy array before the reshape operation.
    """
    # Cast X to a numpy array first to guarantee .reshape() works
    sums = np.sum(np.asarray(X), axis=1).reshape(-1, 1)
    return sums

# This function is the "recipe" for our feature engineering.
# It takes in configuration lists and outputs a ColumnTransformer that can be fitted
# and applied to data later in the pipeline.
# By keeping it separate from any data, we prevent leakage and ensure the same transformations
# are applied consistently to both training and test data.


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    binary_sum_cols: Optional[List[str]] = None,
    n_bins: int = 3,
) -> ColumnTransformer:
    """
    Inputs:
    - quantile_bin_cols: List of numeric column names to be bucketed into quantiles.
    - categorical_onehot_cols: List of categorical column names to be one-hot encoded.
    - numeric_passthrough_cols: List of already-engineered numeric columns to leave untouched.
    - binary_sum_cols: List of binary flag columns to be summed together into a single feature.
    - n_bins: The number of quantile buckets to create.
    Outputs:
    - preprocessor: A scikit-learn ColumnTransformer object representing the recipe.

    Why this contract matters for reliable ML delivery:
    - By taking lists of strings instead of a DataFrame, this function cannot accidentally fit or transform data. It strictly returns the "rules of the game" for the pipeline to execute later.
    """
    print("[features.get_feature_preprocessor] Building feature recipe from configuration")

    if n_bins < 2:
        raise ValueError("Fatal: n_bins must be >= 2 for quantile binning.")

    # Ensure we have empty lists instead of None to prevent iteration errors
    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []
    binary_sum_cols = binary_sum_cols or []

    # Fail-fast: If the recipe is entirely empty, the pipeline will break later anyway.
    if not (quantile_bin_cols or categorical_onehot_cols or numeric_passthrough_cols or binary_sum_cols):
        raise ValueError(
            "Fatal: No feature columns configured for the preprocessor.")

    transformers = []

    # Replacing pd.qcut() and pd.get_dummies()
    # In a notebook, you might use pd.qcut() to create buckets like 'rx_ds_bucket_Q4'.
    # In MLOps, we use KBinsDiscretizer. It learns the exact bin edges on X_train
    # and enforces those exact same edges on X_test, preventing data leakage.
    if quantile_bin_cols:
        quantile_binner = KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",  # Dense output for easier classroom debugging
            strategy="quantile",
        )
        transformers.append(
            ("quantile_bins", quantile_binner, quantile_bin_cols))

    # Handling Categoricals
    # We use handle_unknown="ignore" so the pipeline doesn't crash if X_test
    # contains a category that wasn't present in X_train.
    if categorical_onehot_cols:
        try:
            # Modern scikit-learn versions (1.2+)
            onehot = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # Fallback for older scikit-learn versions to prevent classroom crashes
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat_ohe", onehot, categorical_onehot_cols))

    # Engineered Features via FunctionTransformer
    # Instead of doing df['binary_sum'] = df[['P_D', 'P_P', 'S_P']].sum(axis=1) in Pandas,
    # we embed it in the Pipeline. The deployed model will now calculate this automatically.
    if binary_sum_cols:
        binary_sum_transformer = FunctionTransformer(
            func=_row_sum_numpy,
            validate=False,
            feature_names_out=lambda self, input_features: np.array([
                                                                    "binary_sum"]),
        )
        transformers.append(
            ("binary_sum", binary_sum_transformer, binary_sum_cols))

    # Handling Clean Features
    if numeric_passthrough_cols:
        transformers.append(
            ("num_pass", "passthrough", numeric_passthrough_cols))

    # The Gatekeeper (remainder="drop")
    # This acts as a strict filter. Any column in the raw data that is NOT explicitly
    # listed in one of our configuration lists above will be silently dropped.
    # This prevents raw ID columns or timestamps from accidentally leaking into the model.
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor
