"""
Data cleaning and preprocessing module.
Handles missing value imputation, type conversions, and identifier removal.
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from src.config import (
    CLEANED_DATA_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    ALL_RAW_FEATURES,
)

logger = logging.getLogger(__name__)


def clean_churn_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Customer Churn dataset:
    1. Deep copy input dataframe.
    2. Convert 'TotalCharges' whitespace strings to NaN, then impute with 0.0
       (statistically and domain justified: all 11 missing values occur in new
       customers with tenure == 0 months who have not yet been billed).
    3. Drop unique customer identifier 'customerID'.
    4. Validate data consistency and verify no missing values remain.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw dataset directly loaded from CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for exploratory analysis or feature engineering.
    """
    df = df_raw.copy()

    # Drop customer identifier if present
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
        logger.info(f"Dropped identifier column '{ID_COLUMN}'.")

    # Clean TotalCharges: convert whitespace to numeric
    if "TotalCharges" in df.columns and df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        missing_count = df["TotalCharges"].isna().sum()
        if missing_count > 0:
            logger.info(
                f"Imputing {missing_count} missing TotalCharges values with 0.0 (tenure=0 customers)."
            )
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Verify no nulls remain
    remaining_nulls = df.isna().sum().sum()
    if remaining_nulls > 0:
        raise ValueError(f"Cleaned dataset still contains {remaining_nulls} missing values.")

    return df


def preprocess_pipeline(
    input_path: Optional[Path] = None, output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Complete pipeline to load raw data, clean it, and save the processed dataset.

    Parameters
    ----------
    input_path : Path, optional
        Path to raw data.
    output_path : Path, optional
        Path where cleaned CSV should be written.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    from src.data.loader import load_raw_data

    raw_df = load_raw_data(input_path)
    cleaned_df = clean_churn_data(raw_df)

    out = output_path or CLEANED_DATA_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out, index=False)
    logger.info(f"Cleaned dataset successfully saved to {out}")

    return cleaned_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess_pipeline()

