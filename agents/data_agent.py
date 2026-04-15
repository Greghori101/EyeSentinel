"""
DataIngestionAgent: loads and merges status.csv, locations.json, users.json, personas.md.
Stores results in FeatureStore; returns a compact metadata summary to the LLM.
"""
from __future__ import annotations
import json
import re
import pandas as pd
from pathlib import Path
from langfuse import observe
from core.config import LevelConfig
from core.feature_store import FeatureStore


class DataIngestionAgent:
    def __init__(self, config: LevelConfig, store: FeatureStore):
        self.config = config
        self.store = store

    @observe(name="data_ingestion")
    def run(self, split: str) -> dict:
        """Load and merge data for 'train' or 'eval' split."""
        if split == "train":
            status_path = self.config.train_status
            loc_path = self.config.train_locations
            users_path = self.config.train_users
            personas_path = self.config.train_personas
        else:
            status_path = self.config.eval_status
            loc_path = self.config.eval_locations
            users_path = self.config.eval_users
            personas_path = self.config.eval_personas

        status_df = self._load_status(status_path)
        users_df = self._load_users(users_path)
        locations_df = self._load_locations(loc_path)
        personas = self._load_personas(personas_path)

        self.store.metadata[f"{split}_users_df"] = users_df
        self.store.metadata[f"{split}_locations_df"] = locations_df
        self.store.metadata[f"{split}_personas"] = personas

        self.store.set_raw(split, status_df)

        return self._summarize(status_df, users_df, personas, split)

    def _load_status(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df

    def _load_users(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _load_locations(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame([data])

    def _load_personas(self, path: Path) -> dict:
        """Parse personas.md into a dict mapping CitizenID → structured fields."""
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
        sections = re.split(r"^## ", text, flags=re.MULTILINE)
        personas = {}
        for section in sections[1:]:
            lines = section.strip().split("\n")
            header = lines[0]
            match = re.match(r"([A-Z0-9]+)\s+-\s+(.+)", header)
            if not match:
                continue
            citizen_id, name = match.group(1), match.group(2).strip()

            age_m = re.search(r"\*\*Age:\*\*\s*(\d+)", section)
            occ_m = re.search(r"\*\*Occupation:\*\*\s*([^|]+)", section)
            mob_m = re.search(r"\*\*Mobility:\*\*\s*(.+?)(?:\n|$)", section)
            health_m = re.search(r"\*\*Health behavior:\*\*\s*(.+?)(?:\n|$)", section)
            social_m = re.search(r"\*\*Social pattern:\*\*\s*(.+?)(?:\n|$)", section)

            personas[citizen_id] = {
                "name": name,
                "age": int(age_m.group(1)) if age_m else None,
                "occupation": occ_m.group(1).strip() if occ_m else "",
                "mobility": mob_m.group(1).strip() if mob_m else "",
                "health_behavior": health_m.group(1).strip() if health_m else "",
                "social_pattern": social_m.group(1).strip() if social_m else "",
                "full_text": section.strip(),
            }
        return personas

    def _summarize(self, status_df: pd.DataFrame, users_df: pd.DataFrame | None, personas: dict, split: str) -> dict:
        citizens = status_df["CitizenID"].unique().tolist()
        event_dist = status_df["EventType"].value_counts().to_dict()
        has_escalated = status_df["EventType"].isin([
            "follow-up assessment", "specialist consultation",
            "emergency visit", "urgent care visit", "hospital admission"
        ])
        citizens_with_escalated = status_df[has_escalated]["CitizenID"].unique().tolist()

        persona_summaries = {
            cid: {
                "age": p["age"],
                "occupation": p["occupation"],
                "mobility": p["mobility"],
                "health_behavior": p["health_behavior"],
                "social_pattern": p["social_pattern"],
            }
            for cid, p in personas.items()
        }

        summary = {
            "split": split,
            "n_events": int(len(status_df)),
            "n_citizens": int(len(citizens)),
            "citizen_ids": citizens,
            "event_type_distribution": {k: int(v) for k, v in event_dist.items()},
            "citizens_with_escalated_events": citizens_with_escalated,
            "has_users_data": users_df is not None and not users_df.empty,
            "has_personas": bool(personas),
            "n_personas_loaded": len(personas),
            "persona_summaries": persona_summaries,
            "pai_range": [float(status_df["PhysicalActivityIndex"].min()),
                          float(status_df["PhysicalActivityIndex"].max())],
            "sqi_range": [float(status_df["SleepQualityIndex"].min()),
                          float(status_df["SleepQualityIndex"].max())],
            "eel_range": [float(status_df["EnvironmentalExposureLevel"].min()),
                          float(status_df["EnvironmentalExposureLevel"].max())],
        }
        return summary
