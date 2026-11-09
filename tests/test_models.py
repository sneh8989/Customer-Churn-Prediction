"""
Tests for trained models and model artifacts.
"""

import pytest
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, BEST_MODEL_PATH
from src.data.loader import load_train_test_data
from src.models.train import get_baseline_models


def test_baseline_models_instantiation():
    models = get_baseline_models()
    expected_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine",
    ]
    for name in expected_models:
        assert name in models
        assert hasattr(models[name], "fit")
        assert hasattr(models[name], "predict")


def test_best_model_loads_and_predicts():
    assert BEST_MODEL_PATH.is_file()
    best_model = joblib.load(BEST_MODEL_PATH)
    assert hasattr(best_model, "predict")
    assert hasattr(best_model, "predict_proba")


def test_best_pipeline_loads_and_predicts_on_raw_input(sample_valid_customer):
    pipeline_path = MODELS_DIR / "best_pipeline.pkl"
    assert pipeline_path.is_file(), "best_pipeline.pkl is missing."
    pipeline = joblib.load(pipeline_path)
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    
    # Test directly on raw dataframe
    raw_df = pd.DataFrame([sample_valid_customer])
    pred = pipeline.predict(raw_df)
    prob = pipeline.predict_proba(raw_df)[:, 1]
    
    assert len(pred) == 1
    assert 0.0 <= prob[0] <= 1.0


def test_model_predictions_on_test_data():
    _, X_test, _, y_test = load_train_test_data()
    best_model = joblib.load(BEST_MODEL_PATH)

    preds = best_model.predict(X_test)
    assert len(preds) == len(X_test)

    # Convert string predictions if legacy
    if isinstance(preds[0], str):
        numeric_preds = np.array([1 if p == "Yes" else 0 for p in preds])
    else:
        numeric_preds = preds

    assert set(np.unique(numeric_preds)).issubset({0, 1})

    probs = best_model.predict_proba(X_test)[:, 1]
    assert len(probs) == len(X_test)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


@pytest.mark.parametrize(
    "model_filename",
    [
        "logistic_regression.pkl",
        "decision_tree.pkl",
        "random_forest.pkl",
        "knn.pkl",
        "naive_bayes.pkl",
        "support_vector_machine.pkl",
        "tuned_logistic_regression.pkl",
        "tuned_decision_tree.pkl",
        "tuned_random_forest.pkl",
        "tuned_svm.pkl",
    ],
)
def test_all_saved_models_exist_and_predict(model_filename):
    model_path = MODELS_DIR / model_filename
    assert model_path.is_file(), f"Model file {model_filename} is missing."
    model = joblib.load(model_path)
    X_train, _, _, _ = load_train_test_data()
    dummy_input = pd.DataFrame(np.zeros((2, 45)), columns=X_train.columns)
    preds = model.predict(dummy_input)
    assert len(preds) == 2
