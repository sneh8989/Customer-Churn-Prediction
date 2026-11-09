"""
Data loading module with cross-platform path resolution and validation.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    RAW_DATA_PATH_LOWER,
    CLEANED_DATA_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    ALL_RAW_FEATURES,
)

logger = logging.getLogger(__name__)


def resolve_raw_data_path(custom_path: Optional[Path] = None) -> Path:
    """
    Resolve the raw dataset path in a case-insensitive, cross-platform manner.

    Parameters
    ----------
    custom_path : Path, optional
        Custom path to check first.

    Returns
    -------
    Path
        Resolved valid path to the raw dataset.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found at any standard location.
    """
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend([RAW_DATA_PATH, RAW_DATA_PATH_LOWER])

    # Also check parent relative
    candidates.append(Path("../data/raw/Customer_Churn.csv"))
    candidates.append(Path("../data/raw/customer_churn.csv"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Raw customer churn dataset not found. Checked: {[str(c) for c in candidates]}"
    )


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw Customer Churn dataset and perform initial schema checks.

    Parameters
    ----------
    data_path : Path, optional
        Explicit path to the raw dataset.

    Returns
    -------
    pd.DataFrame
        Loaded raw dataframe.
    """
    path = resolve_raw_data_path(data_path)
    logger.info(f"Loading raw dataset from {path}")
    df = pd.read_csv(path)

    # Basic schema sanity checks
    expected_cols = [ID_COLUMN] + ALL_RAW_FEATURES + [TARGET_COLUMN]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw dataset is missing expected columns: {missing_cols}")

    return df


def load_processed_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the cleaned processed dataset.

    Parameters
    ----------
    data_path : Path, optional
        Explicit path to the cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Loaded cleaned dataframe.
    """
    path = data_path or CLEANED_DATA_PATH
    if not Path(path).is_file():
        # Fallback to parent path if called from notebook
        parent_candidate = Path("../data/processed/churn_cleaned.csv")
        if parent_candidate.is_file():
            path = parent_candidate
        else:
            raise FileNotFoundError(f"Cleaned dataset not found at {path}")

    return pd.read_csv(path)


def load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load processed training and testing datasets.
    Maps string targets ('No'/'Yes') to binary integers (0/1) for pipeline consistency.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    x_train_path = X_TRAIN_PATH if X_TRAIN_PATH.is_file() else Path("../data/processed/X_train.csv")
    x_test_path = X_TEST_PATH if X_TEST_PATH.is_file() else Path("../data/processed/X_test.csv")
    y_train_path = Y_TRAIN_PATH if Y_TRAIN_PATH.is_file() else Path("../data/processed/y_train.csv")
    y_test_path = Y_TEST_PATH if Y_TEST_PATH.is_file() else Path("../data/processed/y_test.csv")

    for p, name in [
        (x_train_path, "X_train"),
        (x_test_path, "X_test"),
        (y_train_path, "y_train"),
        (y_test_path, "y_test"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"Processed file '{name}' not found at {p}")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train_raw = pd.read_csv(y_train_path).squeeze()
    y_test_raw = pd.read_csv(y_test_path).squeeze()

    # Consistently map targets to 0/1
    if y_train_raw.dtype == object:
        y_train = y_train_raw.map({"No": 0, "Yes": 1}).astype(int)
    else:
        y_train = y_train_raw.astype(int)

    if y_test_raw.dtype == object:
        y_test = y_test_raw.map({"No": 0, "Yes": 1}).astype(int)
    else:
        y_test = y_test_raw.astype(int)

    return X_train, X_test, y_train, y_test

