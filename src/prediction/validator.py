"""
Customer input validation module.
Verifies column presence, data types, and value constraints prior to model inference.
"""

import logging
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_RAW_FEATURES,
)

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when customer input data fails validation checks."""
    pass


VALID_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"No phone service", "No", "Yes"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"No", "Yes", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"No", "Yes", "No internet service"},
    "TechSupport": {"No", "Yes", "No internet service"},
    "StreamingTV": {"No", "Yes", "No internet service"},
    "StreamingMovies": {"No", "Yes", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


def validate_customer_input(
    data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Validate raw customer input data and convert to standardized DataFrame.

    Parameters
    ----------
    data : Dict, List[Dict], or pd.DataFrame
        Raw customer record(s).

    Returns
    -------
    pd.DataFrame
        Validated, cleaned DataFrame ready for preprocessor transformation.

    Raises
    ------
    ValidationError
        If input is empty, missing required columns, or contains invalid numerical values.
    """
    if data is None:
        raise ValidationError("Input data cannot be None.")

    if isinstance(data, dict):
        if not data:
            raise ValidationError("Input dictionary cannot be empty.")
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValidationError("Input list cannot be empty.")
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValidationError("Input DataFrame cannot be empty.")
        df = data.copy()
    else:
        raise ValidationError(f"Unsupported input type: {type(data)}.")

    # Drop customerID if passed
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Check for missing required columns
    missing = [c for c in ALL_RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValidationError(f"Input is missing required columns: {missing}")

    # Process and validate numerical features
    for num_col in NUMERICAL_FEATURES:
        # Convert string representations if necessary
        if not pd.api.types.is_numeric_dtype(df[num_col]):
            df[num_col] = pd.to_numeric(df[num_col].astype(str).str.strip(), errors="coerce")
        else:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        # Fill TotalCharges if tenure is 0 and TotalCharges is NaN
        if num_col == "TotalCharges":
            mask = df["tenure"] == 0
            df.loc[mask & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

        # Check for remaining nulls in numerical columns
        if df[num_col].isna().any():
            raise ValidationError(f"Numerical feature '{num_col}' contains invalid or null values.")

        # Range checks
        if (df[num_col] < 0).any():
            raise ValidationError(f"Numerical feature '{num_col}' cannot be negative.")

    if not df["SeniorCitizen"].isin([0, 1]).all():
        raise ValidationError("SeniorCitizen must take values 0 or 1.")

    # Warn or clean categorical features
    for cat_col in CATEGORICAL_FEATURES:
        df[cat_col] = df[cat_col].astype(str).str.strip()
        allowed = VALID_CATEGORIES.get(cat_col, set())
        invalid_mask = ~df[cat_col].isin(allowed)
        if invalid_mask.any():
            invalid_vals = df.loc[invalid_mask, cat_col].unique().tolist()
            logger.warning(
                f"Column '{cat_col}' received unexpected category: {invalid_vals}. "
                "Will be handled by preprocessor OneHotEncoder(handle_unknown='ignore')."
            )

    # Reorder columns to match standard schema
    ordered_df = df[ALL_RAW_FEATURES].copy()
    return ordered_df
