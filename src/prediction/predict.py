"""
Inference module for individual and batch customer churn predictions.
Combines input validation, preprocessing, probability calculation, and risk tiering.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import joblib
import pandas as pd
import numpy as np

from src.config import (
    PREPROCESSOR_PATH,
    BEST_MODEL_PATH,
    RISK_THRESHOLDS,
)
from src.prediction.validator import validate_customer_input

logger = logging.getLogger(__name__)


@dataclass
class ChurnPredictionResult:
    """Structured container for prediction output and business guidance."""
    customer_index: int
    prediction: str
    churn_probability: float
    risk_level: str
    recommendation: str

    @property
    def churn_probability_pct(self) -> str:
        return f"{self.churn_probability * 100:.2f}%"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_index": self.customer_index,
            "prediction": self.prediction,
            "churn_probability": round(self.churn_probability, 4),
            "churn_probability_pct": self.churn_probability_pct,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
        }


def assign_risk_and_recommendation(probability: float) -> Tuple[str, str]:
    """
    Assign risk tier and business retention action based on churn probability.

    Parameters
    ----------
    probability : float
        Model output churn probability [0.0, 1.0].

    Returns
    -------
    Tuple[str, str]
        (risk_level, recommendation)
    """
    low_th = RISK_THRESHOLDS["low"]
    high_th = RISK_THRESHOLDS["high"]

    if probability < low_th:
        risk = "Low Risk"
        action = (
            "Customer shows strong loyalty. Maintain standard customer experience "
            "without discounting. Eligible for premium add-on upsells."
        )
    elif probability < high_th:
        risk = "Medium Risk"
        action = (
            "Customer exhibits emerging churn signals. Conduct a satisfaction check-in, "
            "verify internet connection reliability, and offer annual plan discount."
        )
    else:
        risk = "High Risk"
        action = (
            "Immediate proactive retention required! Assign retention specialist, "
            "offer long-term contract pricing incentive, and review recent technical support tickets."
        )

    return risk, action


def load_artifacts(
    model_path: Optional[Path] = None, preprocessor_path: Optional[Path] = None
) -> Tuple[Any, Any]:
    """Load serialized model and preprocessor artifacts."""
    m_path = Path(model_path or BEST_MODEL_PATH)
    p_path = Path(preprocessor_path or PREPROCESSOR_PATH)

    if not m_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {m_path}")
    if not p_path.is_file():
        raise FileNotFoundError(f"Preprocessor artifact not found at {p_path}")

    model = joblib.load(m_path)
    preprocessor = joblib.load(p_path)
    return model, preprocessor


def predict_churn(
    customer_data: Dict[str, Any],
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    decision_threshold: float = 0.50,
) -> ChurnPredictionResult:
    """
    Predict churn for a single customer record.

    Parameters
    ----------
    customer_data : Dict[str, Any]
        Raw customer record with demographic, service, and contract details.
    model : Any, optional
        Preloaded model instance (loads from disk if None).
    preprocessor : Any, optional
        Preloaded preprocessor instance (loads from disk if None).
    decision_threshold : float, default=0.50
        Probability threshold for classifying customer as 'Churn'.

    Returns
    -------
    ChurnPredictionResult
        Detailed prediction outcome.
    """
    results = predict_batch(
        [customer_data],
        model=model,
        preprocessor=preprocessor,
        decision_threshold=decision_threshold,
    )
    return results[0]


def predict_batch(
    customers: Union[pd.DataFrame, List[Dict[str, Any]]],
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    decision_threshold: float = 0.50,
) -> List[ChurnPredictionResult]:
    """
    Predict churn for a batch of customer records.

    Parameters
    ----------
    customers : pd.DataFrame or List[Dict[str, Any]]
        Multiple raw customer records.
    model : Any, optional
        Trained model instance.
    preprocessor : Any, optional
        Fitted ColumnTransformer instance.
    decision_threshold : float, default=0.50
        Probability threshold for classifying as 'Churn'.

    Returns
    -------
    List[ChurnPredictionResult]
        List of predictions and recommendations for each record.
    """
    if model is None or preprocessor is None:
        loaded_model, loaded_preprocessor = load_artifacts()
        model = model or loaded_model
        preprocessor = preprocessor or loaded_preprocessor

    # 1. Input validation & schema sanitization
    clean_df = validate_customer_input(customers)

    # 2. Preprocessing & feature transformation
    X_proc = preprocessor.transform(clean_df)
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
        X_proc = pd.DataFrame(X_proc, columns=feature_names)

    # 3. Probability calculation
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else (classes.index("Yes") if "Yes" in classes else 1)
        probabilities = model.predict_proba(X_proc)[:, pos_idx]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_proc)
        # Apply sigmoid to decision function
        probabilities = 1.0 / (1.0 + np.exp(-scores))
    else:
        raw_preds = model.predict(X_proc)
        probabilities = np.array([1.0 if p in (1, "Yes") else 0.0 for p in raw_preds])

    # 4. Binary prediction and risk classification
    results = []
    for idx, prob in enumerate(probabilities):
        label = "Churn" if prob >= decision_threshold else "No Churn"
        risk, recommendation = assign_risk_and_recommendation(float(prob))
        results.append(
            ChurnPredictionResult(
                customer_index=idx,
                prediction=label,
                churn_probability=float(prob),
                risk_level=risk,
                recommendation=recommendation,
            )
        )

    return results


if __name__ == "__main__":
    sample_customer = {
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
    result = predict_churn(sample_customer)
    print("Prediction Result:")
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")
