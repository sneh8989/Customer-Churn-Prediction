"""
Central configuration for the Customer Churn Prediction project.
Provides centralized path management, feature definitions, and model hyperparameters.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Root directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "Customer_Churn.csv"
RAW_DATA_PATH_LOWER = RAW_DATA_DIR / "customer_churn.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "churn_cleaned.csv"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"
SAMPLE_CUSTOMERS_PATH = PROCESSED_DATA_DIR / "sample_new_customers.csv"

# Models and artifacts paths
MODELS_DIR = BASE_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
BEST_PIPELINE_PATH = MODELS_DIR / "best_pipeline.pkl"

# Reports and figures paths
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_TRAINING_RESULTS_PATH = REPORTS_DIR / "model_training_results.csv"
MODEL_EVALUATION_RESULTS_PATH = REPORTS_DIR / "model_evaluation_results.csv"
HYPERPARAMETER_TUNING_RESULTS_PATH = REPORTS_DIR / "hyperparameter_tuning_results.csv"
FINAL_MODEL_COMPARISON_PATH = REPORTS_DIR / "final_model_comparison.csv"
CROSS_VALIDATION_RESULTS_PATH = REPORTS_DIR / "cross_validation_results.csv"

# Target and feature schema
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_FEATURES: List[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_RAW_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Risk classification thresholds
RISK_THRESHOLDS: Dict[str, float] = {
    "low": 0.30,
    "high": 0.60,
}

# General project settings
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

