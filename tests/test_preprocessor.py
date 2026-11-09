"""
Tests for data cleaning and preprocessing logic.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import clean_churn_data


def test_clean_churn_data_drops_customer_id(sample_raw_dataframe):
    cleaned = clean_churn_data(sample_raw_dataframe)
    assert "customerID" not in cleaned.columns
    assert len(cleaned) == len(sample_raw_dataframe)


def test_clean_churn_data_handles_blank_total_charges():
    # Synthetic record where tenure is 0 and TotalCharges is whitespace
    raw = pd.DataFrame([{
        "customerID": "0000-NEW",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.0,
        "TotalCharges": "   ",
        "Churn": "No",
    }])
    cleaned = clean_churn_data(raw)
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_clean_churn_data_converts_total_charges_to_float():
    raw = pd.DataFrame([{
        "customerID": "1111-STR",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 25.5,
        "TotalCharges": "51.0",
        "Churn": "No",
    }])
    cleaned = clean_churn_data(raw)
    assert cleaned["TotalCharges"].dtype == np.float64
    assert cleaned["TotalCharges"].iloc[0] == 51.0

