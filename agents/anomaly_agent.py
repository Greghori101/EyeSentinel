from __future__ import annotations

import numpy as np
import pandas as pd

from core.feature_store import DatasetBundle, FeatureStore


class AnomalyAgent:
    def __init__(self, store: FeatureStore):
        self.store = store

    def run(self, bundle: DatasetBundle, merged: pd.DataFrame) -> pd.DataFrame:
        frame = merged.copy()
        graph = self._graph_features(bundle.transactions)
        frame = frame.merge(graph, on="transaction_id", how="left").fillna(0.0)

        numeric = (
            frame.drop(columns=["transaction_id"])
            .select_dtypes(include=["number", "bool"])
            .copy()
        )
        numeric = numeric.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        if len(frame) >= 3:
            isolation_score = self._unsupervised_score(numeric)
        else:
            isolation_score = np.zeros(len(frame), dtype=float)

        rule_score = self._build_rule_score(frame)
        anomaly_score = 0.55 * rule_score + 0.45 * isolation_score

        result = pd.DataFrame(
            {
                "transaction_id": frame["transaction_id"],
                "rule_score": rule_score,
                "isolation_score": isolation_score,
                "graph_score": frame["graph_score"],
                "anomaly_score": anomaly_score,
            }
        )
        self.store.anomaly_scores = result
        return result

    def _unsupervised_score(self, numeric: pd.DataFrame) -> np.ndarray:
        normalized_columns: list[np.ndarray] = []
        for column in numeric.columns:
            normalized_columns.append(
                self._normalize(numeric[column].to_numpy(dtype=float))
            )

        matrix = (
            np.column_stack(normalized_columns)
            if normalized_columns
            else np.zeros((len(numeric), 1))
        )
        top_k = min(5, matrix.shape[1])
        top_feature_mean = np.sort(matrix, axis=1)[:, -top_k:].mean(axis=1)
        overall_mean = matrix.mean(axis=1)
        return 0.6 * top_feature_mean + 0.4 * overall_mean

    def _graph_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        tx = transactions.copy().sort_values("timestamp")
        sender_degree = tx.groupby("sender_id")["recipient_id"].nunique().to_dict()
        recipient_degree = tx.groupby("recipient_id")["sender_id"].nunique().to_dict()
        pair_counts = tx.groupby(["sender_id", "recipient_id"]).size().to_dict()

        records: list[dict] = []
        for row in tx.itertuples(index=False):
            out_degree = float(sender_degree.get(row.sender_id, 0))
            in_degree = float(recipient_degree.get(row.recipient_id, 0))
            pair_count = float(pair_counts.get((row.sender_id, row.recipient_id), 0))
            graph_score = (
                min(
                    (out_degree / 10.0)
                    + (in_degree / 10.0)
                    + (1.0 / max(pair_count, 1.0)),
                    3.0,
                )
                / 3.0
            )
            records.append(
                {
                    "transaction_id": row.transaction_id,
                    "sender_out_degree": out_degree,
                    "recipient_in_degree": in_degree,
                    "pair_count": pair_count,
                    "graph_score": graph_score,
                }
            )
        return pd.DataFrame(records)

    def _build_rule_score(self, frame: pd.DataFrame) -> np.ndarray:
        components = pd.DataFrame(index=frame.index)
        components["amount"] = self._normalize(frame.get("log_amount", 0.0))
        components["amount_to_salary"] = self._normalize(
            frame.get("amount_to_salary", 0.0)
        )
        components["balance_pressure"] = self._normalize(
            frame.get("balance_pressure", 0.0)
        )
        components["amount_dev"] = self._normalize(
            frame.get("sender_amount_deviation", 0.0)
        )
        components["velocity"] = self._normalize(frame.get("sender_rolling_24h", 0.0))
        components["type_rarity"] = frame.get("sender_type_rarity", 0.0).clip(0, 1)
        components["method_rarity"] = frame.get("sender_method_rarity", 0.0).clip(0, 1)
        components["new_recipient"] = frame.get("recipient_is_new", 0.0).clip(0, 1)
        components["night"] = frame.get("is_night", 0.0).clip(0, 1)
        components["description"] = frame.get("description_risk", 0.0).clip(0, 1)
        components["no_iban_transfer"] = frame.get("transfer_without_iban", 0.0).clip(
            0, 1
        )
        components["missing_location"] = frame.get(
            "inperson_without_location", 0.0
        ).clip(0, 1)
        components["phishing"] = frame.get("sender_recent_phishing_30d", 0.0).clip(0, 1)
        components["vulnerability"] = frame.get("sender_vulnerability", 0.0).clip(0, 1)
        components["home_gap"] = self._normalize(
            frame.get("sender_recent_gap_hours", 0.0)
        )
        components["distance"] = self._normalize(
            frame.get("sender_recent_distance_from_home", 0.0)
        )
        components["audio"] = self._normalize(frame.get("sender_audio_30d", 0.0))
        components["graph"] = frame.get("graph_score", 0.0).clip(0, 1)
        weights = np.array(
            [
                0.08,
                0.08,
                0.05,
                0.12,
                0.08,
                0.07,
                0.05,
                0.05,
                0.05,
                0.06,
                0.06,
                0.06,
                0.12,
                0.08,
                0.06,
                0.05,
                0.04,
                0.09,
            ]
        )
        weighted = components.to_numpy(dtype=float) * weights
        return weighted.sum(axis=1)

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
