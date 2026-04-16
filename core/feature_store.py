from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.config import ScenarioConfig


@dataclass
class DatasetBundle:
    config: ScenarioConfig
    transactions: pd.DataFrame
    users: pd.DataFrame
    locations: pd.DataFrame
    sms: list[dict[str, Any]]
    mails: list[dict[str, Any]]
    audio_events: pd.DataFrame
    actor_directory: pd.DataFrame


@dataclass
class FeatureStore:
    bundle: DatasetBundle | None = None
    feature_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    merged_features: pd.DataFrame | None = None
    anomaly_scores: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    reference_profile: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_bundle(self, bundle: DatasetBundle) -> None:
        self.bundle = bundle

    def set_features(self, name: str, frame: pd.DataFrame) -> None:
        self.feature_frames[name] = frame

    def get_features(self, name: str) -> pd.DataFrame:
        return self.feature_frames[name]

    def merge_features(self, key: str = "transaction_id") -> pd.DataFrame:
        if not self.feature_frames:
            raise ValueError("No feature frames available to merge.")

        frames = list(self.feature_frames.values())
        merged = frames[0].copy()
        for frame in frames[1:]:
            merged = merged.merge(frame, on=key, how="left")

        numeric_columns = merged.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        if numeric_columns:
            merged[numeric_columns] = merged[numeric_columns].fillna(0.0)
        merged = merged.fillna("")
        self.merged_features = merged
        return merged
