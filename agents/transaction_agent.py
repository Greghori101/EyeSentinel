from __future__ import annotations

from collections import Counter, deque

import numpy as np
import pandas as pd

from core.feature_store import DatasetBundle, FeatureStore


class TransactionAgent:
    def __init__(self, store: FeatureStore):
        self.store = store

    def run(
        self,
        bundle: DatasetBundle,
        reference_profile: dict | None = None,
    ) -> pd.DataFrame:
        tx = bundle.transactions.copy().sort_values("timestamp").reset_index(drop=True)
        directory = bundle.actor_directory[["actor_id", "salary", "age"]].copy()
        sender_directory = directory.rename(
            columns={
                "actor_id": "sender_id",
                "salary": "sender_salary",
                "age": "sender_age",
            }
        )
        tx = tx.merge(sender_directory, on="sender_id", how="left")

        sender_features = self._build_sender_history_features(tx)
        merged = tx.merge(sender_features, on="transaction_id", how="left")

        merged["hour"] = merged["timestamp"].dt.hour.fillna(0).astype(int)
        merged["weekday"] = merged["timestamp"].dt.weekday.fillna(0).astype(int)
        merged["is_night"] = merged["hour"].isin([0, 1, 2, 3, 4, 5, 23]).astype(int)
        merged["is_weekend"] = merged["weekday"].isin([5, 6]).astype(int)
        merged["has_location"] = merged["location"].fillna("").ne("").astype(int)
        merged["has_sender_iban"] = merged["sender_iban"].fillna("").ne("").astype(int)
        merged["has_recipient_iban"] = (
            merged["recipient_iban"].fillna("").ne("").astype(int)
        )
        merged["log_amount"] = np.log1p(merged["amount"].clip(lower=0))
        merged["balance_pressure"] = merged["amount"] / (
            merged["amount"] + merged["balance_after"].abs() + 1.0
        )
        merged["amount_to_salary"] = merged["amount"] / (
            merged["sender_salary"].fillna(0) + 1.0
        )
        merged["type_baseline_risk"] = merged["transaction_type"].map(
            lambda value: self._baseline_risk(
                value,
                reference_profile,
                "transaction_type_frequency",
            )
        )
        merged["method_baseline_risk"] = merged["payment_method"].map(
            lambda value: self._baseline_risk(
                value,
                reference_profile,
                "payment_method_frequency",
            )
        )
        merged["is_round_amount"] = (merged["amount"] % 10 == 0).astype(int)
        merged["description_risk"] = (
            merged["description"].fillna("").map(self._description_risk)
        )
        merged["transfer_without_iban"] = (
            (merged["transaction_type"] == "bank transfer")
            & (merged["has_sender_iban"] == 0)
        ).astype(int)
        merged["inperson_without_location"] = (
            (merged["transaction_type"] == "in-person payment")
            & (merged["has_location"] == 0)
        ).astype(int)
        merged["recipient_is_new"] = merged["recipient_is_new"].fillna(1.0)
        merged["sender_rolling_24h"] = merged["sender_rolling_24h"].fillna(0.0)
        merged["sender_prev_count"] = merged["sender_prev_count"].fillna(0.0)
        merged["sender_type_rarity"] = merged["sender_type_rarity"].fillna(1.0)
        merged["sender_method_rarity"] = merged["sender_method_rarity"].fillna(1.0)
        merged["sender_amount_deviation"] = merged["sender_amount_deviation"].fillna(
            0.0
        )

        columns = [
            "transaction_id",
            "sender_id",
            "recipient_id",
            "transaction_type",
            "payment_method",
            "amount",
            "log_amount",
            "balance_pressure",
            "amount_to_salary",
            "type_baseline_risk",
            "method_baseline_risk",
            "is_round_amount",
            "hour",
            "weekday",
            "is_night",
            "is_weekend",
            "has_location",
            "has_sender_iban",
            "has_recipient_iban",
            "description_risk",
            "transfer_without_iban",
            "inperson_without_location",
            "sender_prev_count",
            "sender_rolling_24h",
            "sender_amount_deviation",
            "sender_type_rarity",
            "sender_method_rarity",
            "recipient_is_new",
        ]
        features = merged[columns].copy().fillna(0.0)
        self.store.set_features("transaction", features)
        return features

    def _build_sender_history_features(self, tx: pd.DataFrame) -> pd.DataFrame:
        records: list[dict] = []
        for _, group in tx.groupby("sender_id", sort=False):
            history_amounts: list[float] = []
            seen_recipients: set[str] = set()
            type_counter: Counter[str] = Counter()
            method_counter: Counter[str] = Counter()
            recent_window: deque[pd.Timestamp] = deque()

            for row in group.sort_values("timestamp").itertuples(index=False):
                current_ts = row.timestamp
                while (
                    recent_window
                    and (current_ts - recent_window[0]).total_seconds() > 24 * 3600
                ):
                    recent_window.popleft()

                prev_count = len(history_amounts)
                median_amount = (
                    float(np.median(history_amounts)) if history_amounts else 0.0
                )
                mad_amount = (
                    float(np.median(np.abs(np.array(history_amounts) - median_amount)))
                    if history_amounts
                    else 0.0
                )
                amount_deviation = 0.0
                if history_amounts and mad_amount > 0:
                    amount_deviation = float(
                        abs(row.amount - median_amount) / (mad_amount + 1.0)
                    )
                elif history_amounts and median_amount > 0:
                    amount_deviation = float(
                        abs(row.amount - median_amount) / (median_amount + 1.0)
                    )

                recipient_is_new = 1 if row.recipient_id not in seen_recipients else 0
                type_rarity = (
                    1.0
                    if prev_count == 0
                    else 1.0 - (type_counter[row.transaction_type] / prev_count)
                )
                method_value = row.payment_method or "<empty>"
                method_rarity = (
                    1.0
                    if prev_count == 0
                    else 1.0 - (method_counter[method_value] / prev_count)
                )

                records.append(
                    {
                        "transaction_id": row.transaction_id,
                        "sender_prev_count": float(prev_count),
                        "sender_rolling_24h": float(len(recent_window)),
                        "sender_amount_deviation": amount_deviation,
                        "recipient_is_new": float(recipient_is_new),
                        "sender_type_rarity": float(type_rarity),
                        "sender_method_rarity": float(method_rarity),
                    }
                )

                history_amounts.append(float(row.amount))
                seen_recipients.add(row.recipient_id)
                type_counter[row.transaction_type] += 1
                method_counter[method_value] += 1
                recent_window.append(current_ts)
        return pd.DataFrame(records)

    @staticmethod
    def _description_risk(text: str) -> float:
        lowered = (text or "").lower()
        keywords = [
            "security",
            "secure",
            "billing",
            "renewal",
            "urgent",
            "gift",
            "voucher",
            "wallet",
            "crypto",
            "support",
            "refund",
            "subscription",
            "marketplace",
        ]
        score = sum(1 for keyword in keywords if keyword in lowered)
        return float(min(score / 4.0, 1.0))

    @staticmethod
    def _baseline_risk(
        value: str, reference_profile: dict | None, profile_key: str
    ) -> float:
        if not reference_profile:
            return 0.0
        mapping = reference_profile.get(profile_key, {})
        frequency = float(mapping.get(value, 0.0))
        return max(0.0, 1.0 - frequency)
