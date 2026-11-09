"""
Tests for prediction pipeline, input validation, and risk classification.
"""

import pytest
import pandas as pd
import numpy as np

from src.prediction.validator import validate_customer_input, ValidationError
from src.prediction.predict import predict_churn, predict_batch, assign_risk_and_recommendation


def test_predict_churn_single_customer(sample_valid_customer):
    result = predict_churn(sample_valid_customer)
    assert result.prediction in ("Churn", "No Churn")
    assert 0.0 <= result.churn_probability <= 1.0
    assert result.risk_level in ("Low Risk", "Medium Risk", "High Risk")
    assert len(result.recommendation) > 10


def test_predict_loyal_vs_churner(sample_valid_customer, sample_loyal_customer):
    res_churner = predict_churn(sample_valid_customer)
    res_loyal = predict_churn(sample_loyal_customer)

    # Loyal long-term customer should have lower churn risk than month-to-month fiber optic
    assert res_loyal.churn_probability < res_churner.churn_probability


def test_predict_batch_customers(sample_valid_customer, sample_loyal_customer):
    batch = [sample_valid_customer, sample_loyal_customer]
    results = predict_batch(batch)
    assert len(results) == 2
    assert results[0].customer_index == 0
    assert results[1].customer_index == 1


def test_risk_assignment_thresholds():
    risk_low, rec_low = assign_risk_and_recommendation(0.15)
    assert risk_low == "Low Risk"

    risk_med, rec_med = assign_risk_and_recommendation(0.45)
    assert risk_med == "Medium Risk"

    risk_high, rec_high = assign_risk_and_recommendation(0.85)
    assert risk_high == "High Risk"


def test_validation_empty_input_raises_error():
    with pytest.raises(ValidationError, match="empty|None"):
        validate_customer_input({})

    with pytest.raises(ValidationError, match="empty|None"):
        validate_customer_input([])

    with pytest.raises(ValidationError, match="empty|None"):
        validate_customer_input(pd.DataFrame())


def test_validation_missing_required_column(sample_valid_customer):
    bad_customer = dict(sample_valid_customer)
    del bad_customer["tenure"]
    with pytest.raises(ValidationError, match="missing required columns"):
        validate_customer_input(bad_customer)


def test_validation_negative_numerical_values(sample_valid_customer):
    bad_customer = dict(sample_valid_customer)
    bad_customer["MonthlyCharges"] = -10.0
    with pytest.raises(ValidationError, match="cannot be negative"):
        validate_customer_input(bad_customer)


def test_validation_handles_whitespace_total_charges(sample_valid_customer):
    # If customer is new (tenure=0) and TotalCharges is whitespace
    new_customer = dict(sample_valid_customer, tenure=0, TotalCharges="   ")
    validated = validate_customer_input(new_customer)
    assert validated["TotalCharges"].iloc[0] == 0.0


def test_unknown_categorical_value_does_not_crash_pipeline(sample_valid_customer):
    # An unseen InternetService category
    novel_customer = dict(sample_valid_customer, InternetService="Satellite Super")
    result = predict_churn(novel_customer)
    assert result.prediction in ("Churn", "No Churn")
    assert 0.0 <= result.churn_probability <= 1.0


def test_decision_threshold_sensitivity(sample_valid_customer):
    # Low threshold forces Churn
    res_sensitive = predict_churn(sample_valid_customer, decision_threshold=0.10)
    assert res_sensitive.prediction == "Churn"

    # Extremely high threshold forces No Churn
    res_conservative = predict_churn(sample_valid_customer, decision_threshold=0.99)
    assert res_conservative.prediction == "No Churn"


def test_oof_threshold_optimization_peak():
    from src.config import REPORTS_DIR
    th_file = REPORTS_DIR / "threshold_optimization_results.csv"
    assert th_file.is_file(), "threshold_optimization_results.csv is missing."
    df = pd.read_csv(th_file)
    assert "Threshold" in df.columns
    assert "OOF F1 Score" in df.columns
    # Peak F1 score on training out-of-fold predictions occurs at threshold 0.35
    best_th = df.loc[df["OOF F1 Score"].idxmax(), "Threshold"]
    assert best_th == 0.35

