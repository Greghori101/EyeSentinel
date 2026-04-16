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
from core.feature_store import FeatureStore


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

    def run(self, config: ScenarioConfig) -> dict:
        ingestion_summary = self.data_agent.run(config)
        bundle = self.store.bundle
        if bundle is None:
            raise RuntimeError("Data bundle was not loaded.")

        with ThreadPoolExecutor(max_workers=3) as executor:
            tx_future = executor.submit(self.transaction_agent.run, bundle)
            behavior_future = executor.submit(self.behavior_agent.run, bundle)
            communication_future = executor.submit(self.communication_agent.run, bundle)
            tx_features = tx_future.result()
            behavior_features = behavior_future.result()
            communication_features = communication_future.result()

        merged = self.store.merge_features()
        anomaly_scores = self.anomaly_agent.run(bundle, merged)
        predictions = self.decision_agent.run(bundle, merged, anomaly_scores)
        output_path = self._write_output(config, predictions)

        return {
            **ingestion_summary,
            "n_transaction_features": int(len(tx_features.columns) - 1),
            "n_behavior_features": int(len(behavior_features.columns) - 1),
            "n_communication_features": int(len(communication_features.columns) - 1),
            "n_flagged_transactions": int(predictions["predicted_fraud"].sum()),
            "output_path": str(output_path),
            "threshold": float(self.store.metadata.get("decision_threshold", 0.0)),
        }

    def _write_output(self, config: ScenarioConfig, predictions: pd.DataFrame) -> Path:
        output_path = config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        flagged = predictions[predictions["predicted_fraud"] == 1]["transaction_id"].tolist()
        with output_path.open("w", encoding="utf-8") as handle:
            for transaction_id in flagged:
                handle.write(f"{transaction_id}\n")
        return output_path
