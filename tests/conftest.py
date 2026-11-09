"""
Pytest fixtures for data loading, preprocessing, model loading, and predictions.
"""

import pytest
import pandas as pd
from typing import Dict, Any


@pytest.fixture
def sample_valid_customer() -> Dict[str, Any]:
    """Return a single representative raw customer record."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 447.50,
    }


@pytest.fixture
def sample_loyal_customer() -> Dict[str, Any]:
    """Return a customer with characteristics of very low churn probability."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.00,
        "TotalCharges": 2925.00,
    }


@pytest.fixture
def sample_raw_dataframe(sample_valid_customer, sample_loyal_customer) -> pd.DataFrame:
    """Return a 2-row raw dataframe including customerID and target Churn."""
    rec1 = dict(sample_valid_customer, customerID="1000-TESTA", Churn="Yes")
    rec2 = dict(sample_loyal_customer, customerID="2000-TESTB", Churn="No")
    return pd.DataFrame([rec1, rec2])

