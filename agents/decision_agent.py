from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.feature_store import DatasetBundle, FeatureStore


class DecisionAgent:
    def __init__(self, store: FeatureStore):
        self.store = store

    def run(
        self,
        _bundle: DatasetBundle,
        merged: pd.DataFrame,
        anomaly_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = merged.merge(anomaly_scores, on="transaction_id", how="left")
        frame["economic_weight"] = np.log1p(frame["amount"].clip(lower=0))
        frame["combo_score"] = self._combo_score(frame)
        frame["final_score"] = (
            (0.72 * frame["anomaly_score"]) + (0.28 * frame["combo_score"])
        ) * frame["economic_weight"]
        percentile = self._choose_percentile(
            len(frame),
            frame["sender_recent_phishing_30d"].fillna(0.0),
        )
        threshold = float(np.percentile(frame["final_score"], percentile))
        predicted = frame["final_score"] >= threshold

        min_keep = max(1, int(math.ceil(len(frame) * 0.08)))
        max_keep = (
            min(len(frame) - 1, max(min_keep, int(math.ceil(len(frame) * 0.16))))
            if len(frame) > 1
            else 1
        )

        ordered = frame.sort_values("final_score", ascending=False).reset_index(
            drop=True
        )
        keep_count = int(predicted.sum())
        if keep_count < min_keep:
            keep_count = min_keep
        elif keep_count > max_keep:
            keep_count = max_keep
        keep_count = max(1, min(keep_count, max(len(frame) - 1, 1)))

        selected = ordered.head(keep_count).copy()
        selected["predicted_fraud"] = 1
        result = frame[["transaction_id", "final_score"]].merge(
            selected[["transaction_id", "predicted_fraud"]],
            on="transaction_id",
            how="left",
        )
        result["predicted_fraud"] = result["predicted_fraud"].fillna(0).astype(int)
        result = result.sort_values("final_score", ascending=False).reset_index(
            drop=True
        )

        self.store.metadata["decision_threshold"] = threshold
        self.store.metadata["selected_count"] = keep_count
        self.store.predictions = result
        return result

    def _combo_score(self, frame: pd.DataFrame) -> np.ndarray:
        amount_pressure = self._normalize(frame.get("amount_to_salary", 0.0))
        distance_risk = self._normalize(frame.get("sender_recent_distance_from_home", 0.0))
        short_link_risk = self._normalize(frame.get("sender_recent_short_links", 0.0))

        combo = (
            0.20 * frame.get("sender_recent_phishing_7d", 0.0).clip(0, 1)
            + 0.10 * frame.get("sender_recent_phishing_30d", 0.0).clip(0, 1)
            + 0.08 * frame.get("recipient_recent_phishing_30d", 0.0).clip(0, 1)
            + 0.08 * frame.get("description_risk", 0.0).clip(0, 1)
            + 0.08 * frame.get("transfer_without_iban", 0.0).clip(0, 1)
            + 0.07 * frame.get("inperson_without_location", 0.0).clip(0, 1)
            + 0.10 * frame.get("recipient_is_new", 0.0).clip(0, 1)
            + 0.08 * frame.get("sender_type_rarity", 0.0).clip(0, 1)
            + 0.06 * frame.get("sender_method_rarity", 0.0).clip(0, 1)
            + 0.08 * amount_pressure
            + 0.06 * frame.get("sender_far_from_home_flag", 0.0).clip(0, 1)
            + 0.05 * distance_risk
            + 0.04 * short_link_risk
            + 0.02 * frame.get("graph_score", 0.0).clip(0, 1)
        )

        stacked_signal = (
            (frame.get("sender_recent_phishing_30d", 0.0).fillna(0.0) >= 0.5)
            & (frame.get("recipient_is_new", 0.0).fillna(0.0) >= 1.0)
            & (amount_pressure >= 0.6)
        ).astype(float)
        combo = combo + (0.10 * stacked_signal)
        return combo.to_numpy(dtype=float)

    @staticmethod
    def _choose_percentile(n_rows: int, phishing_signal: pd.Series) -> float:
        if n_rows <= 100:
            base = 82.0
        elif n_rows <= 500:
            base = 85.0
        else:
            base = 88.0
        if float(phishing_signal.max()) >= 0.8:
            base -= 2.0
        return float(min(max(base, 75.0), 92.0))

    @staticmethod
    def _normalize(values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr
        low = np.nanpercentile(arr, 5)
        high = np.nanpercentile(arr, 95)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return np.zeros_like(arr)
        return np.clip((arr - low) / (high - low), 0.0, 1.0)
