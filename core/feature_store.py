"""
In-memory store shared between all agents during a single pipeline run.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class FeatureStore:
    # Raw dataframes (indexed by split: "train" or "eval")
    _raw: dict[str, pd.DataFrame] = field(default_factory=dict)
    _features: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Labels derived from training data
    train_labels: pd.Series | None = None

    # Trained model
    model: Any = None

    # Decision threshold chosen by orchestrator
    threshold: float = 0.5

    # Metadata accumulated during the run
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_raw(self, split: str, df: pd.DataFrame) -> None:
        self._raw[split] = df

    def get_raw(self, split: str) -> pd.DataFrame:
        return self._raw[split]

    def set_features(self, split: str, df: pd.DataFrame) -> None:
        self._features[split] = df

    def get_features(self, split: str) -> pd.DataFrame:
        return self._features[split]

    def has_raw(self, split: str) -> bool:
        return split in self._raw

    def has_features(self, split: str) -> bool:
        return split in self._features
