"""
Hyperparameter tuning module using GridSearchCV.
Systematically tunes models, evaluates test set performance, and serializes tuned estimators.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from src.config import (
    MODELS_DIR,
    HYPERPARAMETER_TUNING_RESULTS_PATH,
    RANDOM_STATE,
    CV_FOLDS,
)
from src.data.loader import load_train_test_data

logger = logging.getLogger(__name__)


def get_tuning_configs() -> Dict[str, Dict[str, Any]]:
    """
    Define models and hyperparameter grids for tuning.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Configuration containing estimator and param_grid.
    """
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE
            ),
            "param_grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"],
            },
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        },
        "Support Vector Machine": {
            "estimator": SVC(
                probability=True, random_state=RANDOM_STATE
            ),
            "param_grid": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            },
        },
    }


def tune_all_models(
    save_models: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute hyperparameter tuning using Stratified 5-Fold Cross-Validation,
    evaluate on independent test set, and serialize best estimators.

    Parameters
    ----------
    save_models : bool, default=True
        Whether to dump tuned model artifacts to the models directory.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (tuning_results_df, tuned_models_dict)
    """
    X_train, X_test, y_train, y_test = load_train_test_data()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    configs = get_tuning_configs()

    tuning_records = []
    tuned_models = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, cfg in configs.items():
        logger.info(f"Starting GridSearchCV for {name}...")
        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        tuned_models[name] = best_model

        # Evaluate best tuned model on independent test set
        y_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_acc = accuracy_score(y_test, y_pred)
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_prob)
        else:
            test_auc = roc_auc_score(y_test, y_pred)

        logger.info(
            f"{name} Best CV F1: {grid.best_score_:.4f} | Test F1: {test_f1:.4f} | Best Params: {grid.best_params_}"
        )

        tuning_records.append({
            "Model": name,
            "Best CV F1 Score": grid.best_score_,
            "Test F1 Score": test_f1,
            "Test Accuracy": test_acc,
            "Test ROC AUC": test_auc,
            "Best Parameters": str(grid.best_params_),
        })

        if save_models:
            slug = name.lower().replace(" ", "_")
            if slug == "support_vector_machine":
                filename = "tuned_svm.pkl"
            else:
                filename = f"tuned_{slug}.pkl"
            out_path = MODELS_DIR / filename
            joblib.dump(best_model, out_path)
            logger.info(f"Saved tuned model to {out_path}")

    tuning_df = pd.DataFrame(tuning_records).sort_values(
        by="Best CV F1 Score", ascending=False
    ).reset_index(drop=True)

    HYPERPARAMETER_TUNING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(HYPERPARAMETER_TUNING_RESULTS_PATH, index=False)
    logger.info("Saved hyperparameter tuning results.")

    return tuning_df, tuned_models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tune_all_models()

