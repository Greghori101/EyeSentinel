"""
ModelTrainingAgent: trains a sklearn classifier on the computed features.
Returns CV F1 and top feature importances so the Orchestrator can decide
whether to retrain with different settings.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from langfuse import observe
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from core.config import LevelConfig
from core.feature_store import FeatureStore


MODELS = {
    "gradient_boosting": lambda w: GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    ),
    "random_forest": lambda w: RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42,
        class_weight="balanced" if w else None
    ),
    "logistic_regression": lambda w: LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight="balanced" if w else None
    ),
    "decision_tree": lambda w: DecisionTreeClassifier(
        max_depth=5, random_state=42,
        class_weight="balanced" if w else None
    ),
}


class ModelTrainingAgent:
    def __init__(self, config: LevelConfig, store: FeatureStore):
        self.config = config
        self.store = store

    @observe(name="model_training")
    def run(self, model_type: str, use_class_weight: bool = True) -> dict:
        """
        Train a classifier using stratified k-fold CV.
        Saves the final model (trained on all data) to FeatureStore.
        """
        features_df = self.store.get_features("train")
        labels = self.store.train_labels

        # Align labels to feature rows
        X = features_df.drop(columns=["CitizenID"]).values
        citizen_ids = features_df["CitizenID"].values
        y = np.array([labels.get(cid, 0) for cid in citizen_ids])

        n_samples = len(y)
        n_pos = int(y.sum())
        n_neg = n_samples - n_pos

        if model_type not in MODELS:
            model_type = "gradient_boosting"

        clf = MODELS[model_type](use_class_weight)

        # CV: use min(5, n_samples) folds, at least 2
        n_splits = min(5, max(2, n_samples))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Out-of-fold predictions for threshold tuning
        try:
            oof_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
            oof_pred = (oof_proba >= 0.5).astype(int)
            cv_f1 = f1_score(y, oof_pred, zero_division=0)
        except Exception:
            # If CV fails (too few samples), just compute train F1
            clf.fit(X, y)
            oof_proba = clf.predict_proba(X)[:, 1]
            cv_f1 = f1_score(y, (oof_proba >= 0.5).astype(int), zero_division=0)

        # Store OOF probas for threshold tuning
        self.store.metadata["oof_proba"] = oof_proba
        self.store.metadata["oof_labels"] = y
        self.store.metadata["feature_names"] = features_df.drop(columns=["CitizenID"]).columns.tolist()

        # Fit final model on all data
        clf.fit(X, y)
        self.store.model = clf

        # Feature importances
        importances = []
        feat_names = self.store.metadata["feature_names"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            top_idx = np.argsort(imp)[::-1][:10]
            importances = [(feat_names[i], round(float(imp[i]), 4)) for i in top_idx]
        elif hasattr(clf, "coef_"):
            coef = np.abs(clf.coef_[0])
            top_idx = np.argsort(coef)[::-1][:10]
            importances = [(feat_names[i], round(float(coef[i]), 4)) for i in top_idx]

        return {
            "model_type": model_type,
            "n_samples": n_samples,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "cv_f1": round(float(cv_f1), 4),
            "top_features": importances,
        }
