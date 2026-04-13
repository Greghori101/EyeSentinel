"""
FeatureEngineeringAgent: computes feature groups from raw data.
The Orchestrator LLM decides which groups to compute based on pattern analysis.
"""
from __future__ import annotations
from langfuse import observe
from core.config import LevelConfig
from core.feature_store import FeatureStore
from ml.features import build_feature_matrix, derive_labels


class FeatureEngineeringAgent:
    def __init__(self, config: LevelConfig, store: FeatureStore):
        self.config = config
        self.store = store

    @observe(name="feature_engineering")
    def run(self, split: str, feature_groups: list[str]) -> dict:
        """
        Compute the requested feature groups for the given split.
        Stores result in FeatureStore.
        """
        raw_df = self.store.get_raw(split)
        users_df = self.store.metadata.get(f"{split}_users_df")

        features_df = build_feature_matrix(raw_df, users_df, feature_groups)
        self.store.set_features(split, features_df)

        # Derive labels for training split
        if split == "train":
            self.store.train_labels = derive_labels(raw_df)

        n_features = len(features_df.columns) - 1  # exclude CitizenID
        warnings = []
        if n_features == 0:
            warnings.append("No features computed — check feature_groups list.")
        if "event_features" not in feature_groups:
            warnings.append("event_features not included — this is the strongest signal.")

        return {
            "split": split,
            "n_citizens": len(features_df),
            "n_features": n_features,
            "feature_groups": feature_groups,
            "columns": features_df.columns.tolist(),
            "warnings": warnings,
            "label_distribution": (
                self.store.train_labels.value_counts().to_dict()
                if split == "train" and self.store.train_labels is not None
                else None
            ),
        }
