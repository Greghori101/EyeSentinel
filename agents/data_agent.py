from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from core.config import ScenarioConfig
from core.feature_store import DatasetBundle, FeatureStore


def normalize_text(value: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        value = ""
    elif not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    return " ".join(value.split())


def normalize_name(value: str) -> str:
    return normalize_text(value).replace("_", " ")


def normalize_key(value: str) -> str:
    value = normalize_text(value).replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    return value.strip("_")


class DataIngestionAgent:
    def __init__(self, store: FeatureStore):
        self.store = store

    def run(self, config: ScenarioConfig) -> dict:
        transactions = pd.read_csv(config.dataset_dir / "transactions.csv")
        transactions.columns = [normalize_key(column) for column in transactions.columns]
        transactions["timestamp"] = pd.to_datetime(
            transactions["timestamp"], errors="coerce"
        )
        transactions["amount"] = pd.to_numeric(
            transactions["amount"], errors="coerce"
        ).fillna(0.0)
        transactions["balance_after"] = pd.to_numeric(
            transactions["balance_after"], errors="coerce"
        ).fillna(0.0)
        transactions = self._normalize_transaction_values(transactions)
        transactions = transactions.sort_values("timestamp").reset_index(drop=True)

        users = pd.DataFrame(
            json.loads((config.dataset_dir / "users.json").read_text(encoding="utf-8"))
        )
        users.columns = [normalize_key(column) for column in users.columns]
        users = self._enrich_users(users)

        locations = pd.DataFrame(
            self._normalize_records(
                json.loads(
                    (config.dataset_dir / "locations.json").read_text(encoding="utf-8")
                )
            )
        )
        locations["timestamp"] = pd.to_datetime(locations["timestamp"], errors="coerce")
        locations = locations.sort_values("timestamp").reset_index(drop=True)

        sms = self._normalize_records(
            json.loads((config.dataset_dir / "sms.json").read_text(encoding="utf-8"))
        )
        mails = self._normalize_records(
            (config.dataset_dir / "mails.json").read_text(encoding="utf-8")
        )
        audio_events = self._load_audio_events(config.dataset_dir / "audio")
        actor_directory = self._build_actor_directory(transactions, users, locations)

        bundle = DatasetBundle(
            config=config,
            transactions=transactions,
            users=users,
            locations=locations,
            sms=sms,
            mails=mails,
            audio_events=audio_events,
            actor_directory=actor_directory,
        )
        self.store.set_bundle(bundle)

        return {
            "dataset": config.scenario_name,
            "split": config.split,
            "n_transactions": int(len(transactions)),
            "n_users": int(len(users)),
            "n_locations": int(len(locations)),
            "n_sms": int(len(sms)),
            "n_mails": int(len(mails)),
            "n_audio_events": int(len(audio_events)),
            "n_resolved_actors": int(len(actor_directory)),
            "output_path": str(config.output_path),
        }

    def _enrich_users(self, users: pd.DataFrame) -> pd.DataFrame:
        if users.empty:
            return users

        users = users.copy()
        users["full_name"] = (
            users["first_name"].fillna("") + " " + users["last_name"].fillna("")
        ).str.strip()
        users["normalized_name"] = users["full_name"].map(normalize_name)
        users["first_name_norm"] = users["first_name"].fillna("").map(normalize_name)
        users["salary"] = pd.to_numeric(users.get("salary"), errors="coerce").fillna(
            0.0
        )
        users["birth_year"] = pd.to_numeric(
            users.get("birth_year"), errors="coerce"
        ).fillna(0.0)
        users["age"] = (2087 - users["birth_year"]).clip(lower=0)
        return users

    def _load_audio_events(self, audio_dir: Path) -> pd.DataFrame:
        rows: list[dict] = []
        if not audio_dir.exists():
            return pd.DataFrame(
                columns=["timestamp", "speaker_name", "speaker_name_norm", "file_path"]
            )

        for path in sorted(audio_dir.glob("*.mp3")):
            stem = path.stem
            if "-" not in stem:
                continue
            ts_part, speaker_part = stem.split("-", 1)
            rows.append(
                {
                    "timestamp": pd.to_datetime(
                        ts_part, format="%Y%m%d_%H%M%S", errors="coerce"
                    ),
                    "speaker_name": speaker_part.replace("_", " "),
                    "speaker_name_norm": normalize_name(speaker_part),
                    "file_path": str(path),
                }
            )
        return pd.DataFrame(rows)

    def _normalize_records(self, payload) -> list[dict]:
        if isinstance(payload, str):
            payload = json.loads(payload)
        normalized: list[dict] = []
        for item in payload or []:
            if isinstance(item, dict):
                normalized.append({normalize_key(str(key)): value for key, value in item.items()})
            else:
                normalized.append({"value": item})
        return normalized

    def _normalize_transaction_values(self, transactions: pd.DataFrame) -> pd.DataFrame:
        transactions = transactions.copy()

        type_map = {
            "transfer": "bank transfer",
            "bank transfer": "bank transfer",
            "bonifico": "bank transfer",
            "in person payment": "in-person payment",
            "in-person payment": "in-person payment",
            "e commerce": "e-commerce",
            "e-commerce": "e-commerce",
            "online purchase": "e-commerce",
            "direct debit": "direct debit",
            "withdrawal": "withdrawal",
            "prelievo": "withdrawal",
        }
        method_map = {
            "debit card": "debit card",
            "mobile phone": "mobile device",
            "mobile device": "mobile device",
            "mobile": "mobile device",
            "smartwatch": "smartwatch",
            "google pay": "googlepay",
            "googlepay": "googlepay",
            "paypal": "paypal",
            "pay pal": "paypal",
            "": "",
        }

        transactions["transaction_type"] = transactions["transaction_type"].fillna("").map(
            lambda value: type_map.get(normalize_text(value), normalize_text(value))
        )
        transactions["payment_method"] = transactions["payment_method"].fillna("").map(
            lambda value: method_map.get(normalize_text(value), normalize_text(value))
        )
        return transactions

    def _build_actor_directory(
        self,
        transactions: pd.DataFrame,
        users: pd.DataFrame,
        locations: pd.DataFrame,
    ) -> pd.DataFrame:
        actor_rows: list[dict] = []
        for row in transactions.itertuples(index=False):
            if getattr(row, "sender_id", ""):
                actor_rows.append(
                    {
                        "actor_id": row.sender_id,
                        "iban": row.sender_iban or "",
                        "role": "sender",
                    }
                )
            if getattr(row, "recipient_id", ""):
                actor_rows.append(
                    {
                        "actor_id": row.recipient_id,
                        "iban": row.recipient_iban or "",
                        "role": "recipient",
                    }
                )

        actor_df = pd.DataFrame(actor_rows)
        if actor_df.empty:
            return pd.DataFrame(
                columns=[
                    "actor_id",
                    "iban",
                    "full_name",
                    "normalized_name",
                    "first_name_norm",
                    "salary",
                    "age",
                    "description",
                    "city",
                    "res_lat",
                    "res_lng",
                ]
            )

        grouped = actor_df.groupby("actor_id")
        directory = (
            grouped.agg(
                canonical_iban=("iban", self._mode_non_empty),
                sender_occurrences=("role", lambda s: int((s == "sender").sum())),
                recipient_occurrences=("role", lambda s: int((s == "recipient").sum())),
            )
            .reset_index()
            .rename(columns={"canonical_iban": "iban"})
        )

        if not users.empty:
            user_columns = [
                "iban",
                "full_name",
                "normalized_name",
                "first_name_norm",
                "salary",
                "age",
                "description",
                "residence",
            ]
            merged = directory.merge(users[user_columns], on="iban", how="left")
        else:
            merged = directory.copy()
            merged["full_name"] = ""
            merged["normalized_name"] = ""
            merged["first_name_norm"] = ""
            merged["salary"] = 0.0
            merged["age"] = 0.0
            merged["description"] = ""
            merged["residence"] = None

        if not locations.empty:
            loc_summary = (
                locations.groupby("biotag")
                .agg(
                    n_locations=("biotag", "size"),
                    unique_cities=("city", pd.Series.nunique),
                    first_location_ts=("timestamp", "min"),
                    last_location_ts=("timestamp", "max"),
                )
                .reset_index()
                .rename(columns={"biotag": "actor_id"})
            )
            merged = merged.merge(loc_summary, on="actor_id", how="left")

        merged["n_locations"] = merged.get("n_locations", 0).fillna(0).astype(int)
        merged["unique_cities"] = merged.get("unique_cities", 0).fillna(0).astype(int)
        merged["city"] = merged["residence"].map(
            lambda r: (r or {}).get("city", "") if isinstance(r, dict) else ""
        )
        merged["res_lat"] = pd.to_numeric(
            merged["residence"].map(
                lambda r: (r or {}).get("lat") if isinstance(r, dict) else None
            ),
            errors="coerce",
        )
        merged["res_lng"] = pd.to_numeric(
            merged["residence"].map(
                lambda r: (r or {}).get("lng") if isinstance(r, dict) else None
            ),
            errors="coerce",
        )
        merged = merged.drop(columns=["residence"], errors="ignore")
        return merged

    @staticmethod
    def _mode_non_empty(values: pd.Series) -> str:
        cleaned = [
            value for value in values if isinstance(value, str) and value.strip()
        ]
        if not cleaned:
            return ""
        return Counter(cleaned).most_common(1)[0][0]
