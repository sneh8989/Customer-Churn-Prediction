"""
Tests for data loading module.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import (
    load_raw_data,
    load_processed_data,
    load_train_test_data,
    resolve_raw_data_path,
)


def test_resolve_raw_data_path_exists():
    path = resolve_raw_data_path()
    assert path.is_file()
    assert "churn" in path.name.lower()


def test_load_raw_data_schema():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7043
    assert df.shape[1] == 21
    assert "customerID" in df.columns
    assert "Churn" in df.columns
    assert "TotalCharges" in df.columns


def test_load_processed_data():
    df = load_processed_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7043
    assert df.shape[1] == 20
    assert "customerID" not in df.columns
    assert "TotalCharges" in df.columns
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_load_train_test_data():
    X_train, X_test, y_train, y_test = load_train_test_data()
    assert len(X_train) == 5634
    assert len(X_test) == 1409
    assert len(y_train) == 5634
    assert len(y_test) == 1409
    assert X_train.shape[1] == 45
    assert X_test.shape[1] == 45
    # Verify target encoding is binary integers
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})

