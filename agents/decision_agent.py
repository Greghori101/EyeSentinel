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
        reference_profile: dict | None = None,
    ) -> pd.DataFrame:
        frame = merged.merge(anomaly_scores, on="transaction_id", how="left")
        frame["economic_weight"] = np.log1p(frame["amount"].clip(lower=0))
        frame["combo_score"] = self._combo_score(frame)
        frame["type_prior"] = frame.get("transaction_type", "").map(self._transaction_type_prior)
        frame["actor_score"] = self._actor_propagation_score(frame)
        frame["final_score"] = (
            (0.58 * frame["anomaly_score"])
            + (0.22 * frame["combo_score"])
            + (0.10 * frame["type_prior"])
            + (0.10 * frame["actor_score"])
        ) * frame["economic_weight"]
        local_percentile = self._choose_percentile(
            len(frame),
            frame["sender_recent_phishing_30d"].fillna(0.0),
        )
        percentile = self._adapted_percentile(local_percentile, reference_profile)
        threshold = float(np.percentile(frame["final_score"], percentile))
        threshold = min(
            threshold,
            self._reference_threshold(reference_profile, fallback=threshold),
        )
        predicted = frame["final_score"] >= threshold
        hard_flags = self._hard_flag(frame)
        predicted = predicted | hard_flags

        min_ratio, max_ratio = self._target_band(reference_profile)
        min_keep = max(1, int(math.ceil(len(frame) * min_ratio)))
        max_keep = (
            min(len(frame) - 1, max(min_keep, int(math.ceil(len(frame) * max_ratio))))
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

    def _target_band(self, reference_profile: dict | None) -> tuple[float, float]:
        if not reference_profile:
            return 0.50, 0.72
        flagged_ratio = float(
            reference_profile.get("decision", {}).get("flagged_ratio", 0.28)
        )
        min_ratio = max(0.45, min(0.65, flagged_ratio * 1.45))
        max_ratio = max(min_ratio, min(0.80, flagged_ratio * 1.85 + 0.06))
        return min_ratio, max_ratio

    def _adapted_percentile(
        self,
        local_percentile: float,
        reference_profile: dict | None,
    ) -> float:
        if not reference_profile:
            return local_percentile
        reference_percentile = float(
            reference_profile.get("decision", {}).get(
                "recommended_percentile", local_percentile
            )
        )
        return min(local_percentile, reference_percentile)

    def _reference_threshold(
        self,
        reference_profile: dict | None,
        fallback: float,
    ) -> float:
        if not reference_profile:
            return fallback
        return float(
            reference_profile.get("decision", {}).get("score_threshold", fallback)
        )

    def _hard_flag(self, frame: pd.DataFrame) -> pd.Series:
        amount_pressure = self._normalize(frame.get("amount_to_salary", 0.0))
        high_amount = self._normalize(frame.get("amount", 0.0))
        distance_risk = self._normalize(
            frame.get("sender_recent_distance_from_home", 0.0)
        )

        phishing_stack = (
            frame.get("sender_recent_phishing_30d", 0.0).fillna(0.0) >= 0.55
        ) & (
            (frame.get("recipient_is_new", 0.0).fillna(0.0) >= 1.0)
            | (frame.get("transfer_without_iban", 0.0).fillna(0.0) >= 1.0)
            | (frame.get("inperson_without_location", 0.0).fillna(0.0) >= 1.0)
        )

        economic_stack = (
            (amount_pressure >= 0.7)
            & (high_amount >= 0.75)
            & (
                (frame.get("sender_type_rarity", 0.0).fillna(0.0) >= 0.75)
                | (frame.get("sender_method_rarity", 0.0).fillna(0.0) >= 0.75)
                | (frame.get("recipient_is_new", 0.0).fillna(0.0) >= 1.0)
            )
        )

        geo_stack = (
            (distance_risk >= 0.8)
            & (frame.get("sender_far_from_home_flag", 0.0).fillna(0.0) >= 1.0)
            & (high_amount >= 0.55)
        )

        description_stack = (frame.get("description_risk", 0.0).fillna(0.0) >= 0.6) & (
            frame.get("sender_recent_short_links", 0.0).fillna(0.0) >= 1.0
        )

        return phishing_stack | economic_stack | geo_stack | description_stack

    def _combo_score(self, frame: pd.DataFrame) -> np.ndarray:
        amount_pressure = self._normalize(frame.get("amount_to_salary", 0.0))
        distance_risk = self._normalize(
            frame.get("sender_recent_distance_from_home", 0.0)
        )
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

    def _actor_propagation_score(self, frame: pd.DataFrame) -> np.ndarray:
        base_signal = (
            0.45 * frame.get("sender_recent_phishing_30d", 0.0).fillna(0.0).clip(0, 1)
            + 0.25 * frame.get("recipient_is_new", 0.0).fillna(0.0).clip(0, 1)
            + 0.20 * frame.get("description_risk", 0.0).fillna(0.0).clip(0, 1)
            + 0.10 * frame.get("graph_score", 0.0).fillna(0.0).clip(0, 1)
        )
        actor_frame = pd.DataFrame(
            {
                "sender_id": frame.get("sender_id", ""),
                "recipient_id": frame.get("recipient_id", ""),
                "signal": base_signal.astype(float),
            }
        )

        sender_signal = actor_frame.groupby("sender_id")["signal"].transform("mean")
        recipient_signal = actor_frame.groupby("recipient_id")["signal"].transform("mean")
        combined = 0.65 * sender_signal + 0.35 * recipient_signal
        return self._normalize(combined.to_numpy(dtype=float))

    @staticmethod
    def _transaction_type_prior(value: str) -> float:
        value = str(value or "")
        priors = {
            "bank transfer": 1.0,
            "withdrawal": 0.9,
            "e-commerce": 0.75,
            "in-person payment": 0.55,
            "direct debit": 0.45,
        }
        return float(priors.get(value, 0.5))

    @staticmethod
    def _choose_percentile(n_rows: int, phishing_signal: pd.Series) -> float:
        if n_rows <= 100:
            base = 55.0
        elif n_rows <= 500:
            base = 58.0
        else:
            base = 60.0
        if float(phishing_signal.max()) >= 0.8:
            base -= 6.0
        return float(min(max(base, 45.0), 82.0))

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
