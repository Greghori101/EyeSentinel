from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from agents.anomaly_agent import AnomalyAgent
from agents.behavior_agent import BehaviorAgent
from agents.communication_agent import CommunicationAgent
from agents.data_agent import DataIngestionAgent
from agents.decision_agent import DecisionAgent
from agents.transaction_agent import TransactionAgent
from core.config import ScenarioConfig
from core.feature_store import DatasetBundle, FeatureStore


class FraudOrchestrator:
    def __init__(self, store: FeatureStore, output_dir: Path):
        self.store = store
        self.output_dir = output_dir
        self.data_agent = DataIngestionAgent(store)
        self.transaction_agent = TransactionAgent(store)
        self.behavior_agent = BehaviorAgent(store)
        self.communication_agent = CommunicationAgent(store)
        self.anomaly_agent = AnomalyAgent(store)
        self.decision_agent = DecisionAgent(store)

    def run(
        self,
        config: ScenarioConfig,
        reference_profile: dict | None = None,
        write_output: bool = True,
    ) -> dict:
        ingestion_summary = self.data_agent.run(config)
        bundle = self.store.bundle
        if bundle is None:
            raise RuntimeError("Data bundle was not loaded.")

        with ThreadPoolExecutor(max_workers=3) as executor:
            tx_future = executor.submit(
                self.transaction_agent.run, bundle, reference_profile
            )
            behavior_future = executor.submit(self.behavior_agent.run, bundle)
            communication_future = executor.submit(
                self.communication_agent.run, bundle, reference_profile
            )
            tx_features = tx_future.result()
            behavior_features = behavior_future.result()
            communication_features = communication_future.result()

        merged = self.store.merge_features()
        anomaly_scores = self.anomaly_agent.run(bundle, merged, reference_profile)
        predictions = self.decision_agent.run(
            bundle, merged, anomaly_scores, reference_profile
        )
        output_path = self._write_output(config, predictions) if write_output else None
        self.store.reference_profile = self._build_reference_profile(
            bundle, merged, predictions
        )

        return {
            **ingestion_summary,
            "n_transaction_features": int(len(tx_features.columns) - 1),
            "n_behavior_features": int(len(behavior_features.columns) - 1),
            "n_communication_features": int(len(communication_features.columns) - 1),
            "n_flagged_transactions": int(predictions["predicted_fraud"].sum()),
            "output_path": str(output_path) if output_path else "",
            "threshold": float(self.store.metadata.get("decision_threshold", 0.0)),
        }

    def _write_output(self, config: ScenarioConfig, predictions: pd.DataFrame) -> Path:
        output_path = config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        flagged = predictions[predictions["predicted_fraud"] == 1][
            "transaction_id"
        ].tolist()
        with output_path.open("w", encoding="utf-8") as handle:
            for transaction_id in flagged:
                handle.write(f"{transaction_id}\n")
        return output_path

    def _build_reference_profile(
        self,
        bundle: DatasetBundle,
        merged: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> dict:
        numeric_columns = merged.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        feature_ranges: dict[str, tuple[float, float]] = {}
        for column in numeric_columns:
            series = pd.to_numeric(merged[column], errors="coerce")
            if series.notna().any():
                feature_ranges[column] = (
                    float(series.quantile(0.05)),
                    float(series.quantile(0.95)),
                )

        transactions = bundle.transactions.copy()
        transaction_type_frequency = (
            transactions["transaction_type"].value_counts(normalize=True).to_dict()
        )
        payment_method_frequency = (
            transactions["payment_method"]
            .fillna("")
            .value_counts(normalize=True)
            .to_dict()
        )
        final_scores = predictions["final_score"].astype(float)
        flagged_ratio = (
            float(predictions["predicted_fraud"].mean()) if len(predictions) else 0.0
        )
        recommended_percentile = max(60.0, min(82.0, 100.0 * (1.0 - flagged_ratio)))

        return {
            "feature_ranges": feature_ranges,
            "transaction_type_frequency": transaction_type_frequency,
            "payment_method_frequency": payment_method_frequency,
            "message_memory": self._build_message_memory(),
            "decision": {
                "score_threshold": float(
                    self.store.metadata.get("decision_threshold", 0.0)
                ),
                "flagged_ratio": flagged_ratio,
                "recommended_percentile": recommended_percentile,
                "score_q70": (
                    float(final_scores.quantile(0.70)) if len(final_scores) else 0.0
                ),
                "score_q80": (
                    float(final_scores.quantile(0.80)) if len(final_scores) else 0.0
                ),
            },
        }

    def _build_message_memory(self) -> list[dict]:
        parsed_messages = self.store.metadata.get("parsed_messages")
        if parsed_messages is None or len(parsed_messages) == 0:
            return []

        frame = parsed_messages.copy()
        if "message_risk" not in frame.columns:
            return []

        frame = frame.sort_values("message_risk", ascending=False)
        frame = frame.drop_duplicates(subset=["normalized_text"])
        frame = frame[frame["message_risk"] >= 0.45].head(120)
        memory: list[dict] = []
        for row in frame.itertuples(index=False):
            memory.append(
                {
                    "text": str(getattr(row, "normalized_text", "")),
                    "score": float(getattr(row, "message_risk", 0.0)),
                    "source": str(getattr(row, "source", "")),
                }
            )
        return memory
