"""
Feature engineering and transformation pipeline.
Constructs leak-free ColumnTransformers, performs stratified splitting,
and serializes preprocessing artifacts.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    CLEANED_DATA_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    PREPROCESSOR_PATH,
    SAMPLE_CUSTOMERS_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def create_preprocessor(include_imputer: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numerical and categorical feature pipelines.

    Parameters
    ----------
    include_imputer : bool, default=True
        Whether to include a SimpleImputer before StandardScaler to safeguard
        against missing values during inference.

    Returns
    -------
    ColumnTransformer
        Unfitted column transformer.
    """
    if include_imputer:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_and_save_features(
    cleaned_data_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Execute end-to-end feature engineering workflow:
    1. Load cleaned data.
    2. Split into features (X) and target (y).
    3. Perform stratified train-test split (80/20).
    4. Fit preprocessing pipeline ONLY on X_train.
    5. Transform X_train and X_test independently (zero data leakage).
    6. Serialize preprocessor and transformed datasets to disk.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]
        (X_train_df, X_test_df, y_train, y_test, fitted_preprocessor)
    """
    from src.data.loader import load_processed_data

    df = load_processed_data(cleaned_data_path)

    X = df.drop(columns=[TARGET_COLUMN])
    y_raw = df[TARGET_COLUMN]

    # Map string target to binary integer
    y = y_raw.map({"No": 0, "Yes": 1}).astype(int)

    # Stratified Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Split shapes: X_train={X_train.shape}, X_test={X_test.shape}")

    # Build and fit preprocessor on training data only
    preprocessor = create_preprocessor(include_imputer=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

    # Ensure output directories exist
    X_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save processed datasets
    X_train_df.to_csv(X_TRAIN_PATH, index=False)
    X_test_df.to_csv(X_TEST_PATH, index=False)
    pd.DataFrame({"Churn": y_train}).to_csv(Y_TRAIN_PATH, index=False)
    pd.DataFrame({"Churn": y_test}).to_csv(Y_TEST_PATH, index=False)

    # Save sample raw customers for testing and demonstration
    sample_customers = X_test.head(10).copy()
    sample_customers.to_csv(SAMPLE_CUSTOMERS_PATH, index=False)

    # Save fitted preprocessor artifact
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Preprocessor serialized to {PREPROCESSOR_PATH}")

    return X_train_df, X_test_df, y_train, y_test, preprocessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_and_save_features()

