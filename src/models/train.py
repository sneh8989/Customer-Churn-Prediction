"""
Model training module.
Trains all six baseline classification algorithms with consistent seeds,
evaluates cross-validation metrics, and serializes trained models.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    MODEL_TRAINING_RESULTS_PATH,
    CROSS_VALIDATION_RESULTS_PATH,
    RANDOM_STATE,
    CV_FOLDS,
)
from src.data.loader import load_train_test_data

logger = logging.getLogger(__name__)


def get_baseline_models() -> Dict[str, Any]:
    """
    Instantiate the six baseline classification models with deterministic random states.
    Note: SVC is initialized with probability=True so predict_proba is supported.

    Returns
    -------
    Dict[str, Any]
        Dictionary of model names and classifier instances.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            probability=True, random_state=RANDOM_STATE
        ),
    }


def train_all_baseline_models(
    save_models: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Train all six baseline models, compute stratified cross-validation performance,
    evaluate initial test accuracy, and save models and training reports.

    Parameters
    ----------
    save_models : bool, default=True
        Whether to dump model artifacts to the models directory.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        (training_results_df, cv_results_df, trained_models_dict)
    """
    X_train, X_test, y_train, y_test = load_train_test_data()

    models = get_baseline_models()
    trained_models = {}
    training_records = []
    cv_records = []

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        logger.info(f"Training baseline model: {name}")

        # Cross-validation on training set
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        cv_records.append({
            "Model": name,
            "CV Accuracy Mean": cv_scores["test_accuracy"].mean(),
            "CV Accuracy Std": cv_scores["test_accuracy"].std(),
            "CV Precision Mean": cv_scores["test_precision"].mean(),
            "CV Recall Mean": cv_scores["test_recall"].mean(),
            "CV F1 Mean": cv_scores["test_f1"].mean(),
            "CV F1 Std": cv_scores["test_f1"].std(),
            "CV ROC AUC Mean": cv_scores["test_roc_auc"].mean(),
        })

        # Fit model on entire training dataset
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate initial accuracy on test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        training_records.append({"Model": name, "Accuracy": acc})

        # Serialize baseline model
        if save_models:
            filename = name.lower().replace(" ", "_") + ".pkl"
            out_path = MODELS_DIR / filename
            joblib.dump(model, out_path)
            logger.info(f"Saved {name} to {out_path}")

    # Build and sort summary DataFrames
    training_df = pd.DataFrame(training_records).sort_values(
        by="Accuracy", ascending=False
    ).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_records).sort_values(
        by="CV F1 Mean", ascending=False
    ).reset_index(drop=True)

    training_df.to_csv(MODEL_TRAINING_RESULTS_PATH, index=False)
    cv_df.to_csv(CROSS_VALIDATION_RESULTS_PATH, index=False)
    logger.info("Saved baseline training and cross-validation reports.")

    return training_df, cv_df, trained_models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_all_baseline_models()

