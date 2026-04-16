from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.feature_store import DatasetBundle, FeatureStore


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


class BehaviorAgent:
    def __init__(self, store: FeatureStore):
        self.store = store

    def run(self, bundle: DatasetBundle) -> pd.DataFrame:
        tx = bundle.transactions.copy().sort_values("timestamp").reset_index(drop=True)
        actor_directory = bundle.actor_directory.copy()
        locations = bundle.locations.copy().sort_values("timestamp").reset_index(drop=True)

        location_summary = self._build_location_summary(locations)
        location_recent = self._build_recent_location_features(tx, locations, actor_directory)
        behavior = tx[["transaction_id", "sender_id", "recipient_id", "location", "transaction_type"]].copy()
        behavior = behavior.merge(
            location_summary.rename(columns={"actor_id": "sender_id"}), on="sender_id", how="left"
        )
        behavior = behavior.merge(location_recent, on="transaction_id", how="left")

        behavior["inperson_location_match"] = behavior.apply(self._inperson_location_match, axis=1)
        behavior["sender_recent_gap_hours"] = behavior["sender_recent_gap_hours"].fillna(999.0)
        behavior["sender_recent_distance_from_home"] = behavior[
            "sender_recent_distance_from_home"
        ].fillna(0.0)
        behavior["sender_unique_cities"] = behavior["sender_unique_cities"].fillna(0.0)
        behavior["sender_max_jump_km"] = behavior["sender_max_jump_km"].fillna(0.0)
        behavior["sender_avg_jump_km"] = behavior["sender_avg_jump_km"].fillna(0.0)
        behavior["sender_recent_gap_flag"] = (behavior["sender_recent_gap_hours"] > 24 * 7).astype(int)
        behavior["sender_far_from_home_flag"] = (
            behavior["sender_recent_distance_from_home"] > 500
        ).astype(int)

        features = behavior[
            [
                "transaction_id",
                "sender_unique_cities",
                "sender_max_jump_km",
                "sender_avg_jump_km",
                "sender_recent_gap_hours",
                "sender_recent_gap_flag",
                "sender_recent_distance_from_home",
                "sender_far_from_home_flag",
                "inperson_location_match",
            ]
        ].copy()
        features = features.fillna(0.0)
        self.store.set_features("behavior", features)
        return features

    def _build_location_summary(self, locations: pd.DataFrame) -> pd.DataFrame:
        records: list[dict] = []
        for actor_id, group in locations.groupby("biotag", sort=False):
            group = group.sort_values("timestamp")
            unique_cities = int(group["city"].nunique())
            jumps: list[float] = []
            prev = None
            for row in group.itertuples(index=False):
                if prev is not None:
                    jumps.append(haversine_km(prev.lat, prev.lng, row.lat, row.lng))
                prev = row
            records.append(
                {
                    "actor_id": actor_id,
                    "sender_unique_cities": unique_cities,
                    "sender_max_jump_km": float(max(jumps) if jumps else 0.0),
                    "sender_avg_jump_km": float(np.mean(jumps) if jumps else 0.0),
                }
            )
        return pd.DataFrame(records)

    def _build_recent_location_features(
        self,
        tx: pd.DataFrame,
        locations: pd.DataFrame,
        actor_directory: pd.DataFrame,
    ) -> pd.DataFrame:
        residence = (
            actor_directory.set_index("actor_id")[["res_lat", "res_lng", "city"]].to_dict("index")
            if not actor_directory.empty
            else {}
        )
        location_groups = {
            actor_id: group.sort_values("timestamp")
            for actor_id, group in locations.groupby("biotag", sort=False)
        }

        records: list[dict] = []
        for row in tx.itertuples(index=False):
            actor_locations = location_groups.get(row.sender_id)
            gap_hours = np.nan
            distance_from_home = np.nan
            recent_city = ""
            if actor_locations is not None and not actor_locations.empty:
                prior = actor_locations[actor_locations["timestamp"] <= row.timestamp]
                if not prior.empty:
                    last_row = prior.iloc[-1]
                    gap_hours = (row.timestamp - last_row["timestamp"]).total_seconds() / 3600.0
                    recent_city = str(last_row.get("city", "") or "")
                    res = residence.get(row.sender_id, {})
                    if (
                        pd.notna(last_row.get("lat"))
                        and pd.notna(last_row.get("lng"))
                        and pd.notna(res.get("res_lat"))
                        and pd.notna(res.get("res_lng"))
                    ):
                        distance_from_home = haversine_km(
                            float(last_row["lat"]),
                            float(last_row["lng"]),
                            float(res["res_lat"]),
                            float(res["res_lng"]),
                        )
            records.append(
                {
                    "transaction_id": row.transaction_id,
                    "sender_recent_gap_hours": gap_hours,
                    "sender_recent_distance_from_home": distance_from_home,
                    "sender_recent_city": recent_city,
                    "sender_home_city": residence.get(row.sender_id, {}).get("city", ""),
                }
            )
        return pd.DataFrame(records)

    @staticmethod
    def _inperson_location_match(row: pd.Series) -> float:
        if row.get("transaction_type") != "in-person payment":
            return 0.0

        location_text = str(row.get("location") or "").lower()
        recent_city = str(row.get("sender_recent_city") or "").lower()
        home_city = str(row.get("sender_home_city") or "").lower()
        if not location_text:
            return 0.0
        if recent_city and recent_city in location_text:
            return 1.0
        if home_city and home_city in location_text:
            return 1.0
        return 0.0
