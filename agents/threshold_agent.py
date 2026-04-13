"""
ThresholdTuningAgent: sweeps decision thresholds on OOF predictions
to find the one that maximizes F1 score.
"""
from __future__ import annotations
import numpy as np
from langfuse import observe
from sklearn.metrics import f1_score, precision_score, recall_score
from core.config import LevelConfig
from core.feature_store import FeatureStore


class ThresholdTuningAgent:
    def __init__(self, config: LevelConfig, store: FeatureStore):
        self.config = config
        self.store = store

    @observe(name="threshold_tuning")
    def run(self) -> dict:
        """
        Sweep thresholds from 0.05 to 0.95 and return the F1-maximizing threshold.
        """
        oof_proba = self.store.metadata.get("oof_proba")
        y = self.store.metadata.get("oof_labels")

        if oof_proba is None or y is None:
            return {"error": "No OOF predictions available. Run train_model first."}

        best_f1 = -1.0
        best_threshold = 0.5
        sweep_results = []

        for t in np.arange(0.05, 0.96, 0.05):
            preds = (oof_proba >= t).astype(int)
            f1 = f1_score(y, preds, zero_division=0)
            prec = precision_score(y, preds, zero_division=0)
            rec = recall_score(y, preds, zero_division=0)
            n_pos = int(preds.sum())
            sweep_results.append({
                "threshold": round(float(t), 2),
                "f1": round(float(f1), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "n_predicted_positive": n_pos,
            })
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        self.store.threshold = best_threshold

        return {
            "optimal_threshold": round(best_threshold, 2),
            "max_f1": round(best_f1, 4),
            "sweep_summary": [r for r in sweep_results if r["f1"] > 0],
            "recommendation": (
                f"Use threshold={best_threshold:.2f} to maximize F1={best_f1:.4f}."
            ),
        }
