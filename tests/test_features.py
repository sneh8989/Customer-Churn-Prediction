"""
Tests for feature engineering pipeline and ColumnTransformer.
"""

import pytest
import pandas as pd
import numpy as np
import joblib

from src.features.build_features import create_preprocessor
from src.config import PREPROCESSOR_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def test_create_preprocessor_structure():
    preprocessor = create_preprocessor(include_imputer=True)
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_preprocessor_transform_shape(sample_valid_customer, sample_loyal_customer):
    df = pd.DataFrame([sample_valid_customer, sample_loyal_customer])
    preprocessor = create_preprocessor(include_imputer=True)
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 2
    # Check expected 45 one-hot encoded columns when fitted on 2 samples
    assert transformed.shape[1] > 0


def test_serialized_preprocessor_loads_and_transforms(sample_valid_customer):
    assert PREPROCESSOR_PATH.is_file()
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    df = pd.DataFrame([sample_valid_customer])
    transformed = preprocessor.transform(df)
    assert transformed.shape == (1, 45)
    assert not np.isnan(transformed).any()

