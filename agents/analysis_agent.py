"""
PatternAnalysisAgent: runs statistical analyses and returns compact summaries.
The Orchestrator LLM uses these summaries to make informed feature/model decisions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from langfuse import observe
from core.config import LevelConfig, ESCALATED_EVENT_TYPES, STANDARD_EVENT_TYPES
from core.feature_store import FeatureStore


class PatternAnalysisAgent:
    def __init__(self, config: LevelConfig, store: FeatureStore):
        self.config = config
        self.store = store

    @observe(name="pattern_analysis")
    def run(self, analysis_type: str) -> dict:
        """
        Run the requested analysis on training data.
        Returns a compact summary dict.
        """
        if analysis_type == "event_type_distribution":
            return self._event_type_distribution()
        elif analysis_type == "biometric_trends":
            return self._biometric_trends()
        elif analysis_type == "label_derivation":
            return self._label_derivation()
        elif analysis_type == "full_summary":
            return self._full_summary()
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}

    def _event_type_distribution(self) -> dict:
        df = self.store.get_raw("train")
        event_counts = df["EventType"].value_counts().to_dict()
        escalated_events = {k: v for k, v in event_counts.items() if k in ESCALATED_EVENT_TYPES}
        standard_events = {k: v for k, v in event_counts.items() if k in STANDARD_EVENT_TYPES}

        citizens_with_escalated = df[df["EventType"].isin(ESCALATED_EVENT_TYPES)]["CitizenID"].unique().tolist()
        citizens_only_standard = [
            c for c in df["CitizenID"].unique()
            if c not in citizens_with_escalated
        ]

        return {
            "analysis": "event_type_distribution",
            "all_event_types": event_counts,
            "standard_event_types": standard_events,
            "escalated_event_types": escalated_events,
            "n_citizens_with_escalated": len(citizens_with_escalated),
            "n_citizens_only_standard": len(citizens_only_standard),
            "citizens_label_1": citizens_with_escalated,
            "citizens_label_0": citizens_only_standard,
            "key_insight": (
                "Citizens with escalated event types (emergency visit, specialist consultation, "
                "follow-up assessment) are at risk (label=1). This is the primary signal."
            ),
        }

    def _biometric_trends(self) -> dict:
        df = self.store.get_raw("train").sort_values("Timestamp")
        escalated_citizens = df[df["EventType"].isin(ESCALATED_EVENT_TYPES)]["CitizenID"].unique()

        summaries = {}
        for col in ["PhysicalActivityIndex", "SleepQualityIndex", "EnvironmentalExposureLevel"]:
            at_risk = df[df["CitizenID"].isin(escalated_citizens)].groupby("CitizenID")[col].apply(list)
            normal = df[~df["CitizenID"].isin(escalated_citizens)].groupby("CitizenID")[col].apply(list)

            def trend_slope(vals):
                if len(vals) < 2:
                    return 0.0
                x = list(range(len(vals)))
                n = len(vals)
                x_mean = sum(x) / n
                y_mean = sum(vals) / n
                num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, vals))
                denom = sum((xi - x_mean) ** 2 for xi in x)
                return num / denom if denom != 0 else 0.0

            at_risk_slopes = [trend_slope(v) for v in at_risk]
            normal_slopes = [trend_slope(v) for v in normal]

            summaries[col] = {
                "at_risk_mean_slope": float(np.mean(at_risk_slopes)) if at_risk_slopes else None,
                "normal_mean_slope": float(np.mean(normal_slopes)) if normal_slopes else None,
            }

        return {
            "analysis": "biometric_trends",
            "biometric_slope_comparison": summaries,
            "key_insight": (
                "At-risk citizens typically show declining PAI and SQI (negative slope) "
                "and increasing EEL (positive slope) over time."
            ),
        }

    def _label_derivation(self) -> dict:
        df = self.store.get_raw("train")
        escalated = df[df["EventType"].isin(ESCALATED_EVENT_TYPES)]["CitizenID"].unique().tolist()
        all_citizens = df["CitizenID"].unique().tolist()
        normal = [c for c in all_citizens if c not in escalated]
        return {
            "analysis": "label_derivation",
            "method": "Citizens with any escalated event type (emergency visit, specialist consultation, follow-up assessment) are classified as label=1.",
            "label_1_citizens": escalated,
            "label_0_citizens": normal,
            "n_label_1": len(escalated),
            "n_label_0": len(normal),
            "class_ratio": round(len(escalated) / len(all_citizens), 3) if all_citizens else 0,
        }

    def _full_summary(self) -> dict:
        event_dist = self._event_type_distribution()
        biometric = self._biometric_trends()
        labels = self._label_derivation()
        df = self.store.get_raw("train")
        return {
            "analysis": "full_summary",
            "n_citizens": int(df["CitizenID"].nunique()),
            "n_events": int(len(df)),
            "event_type_analysis": event_dist,
            "biometric_trends": biometric,
            "label_derivation": labels,
            "recommended_features": ["event_features", "biometric_features", "temporal_features"],
            "recommended_model": "gradient_boosting",
        }
