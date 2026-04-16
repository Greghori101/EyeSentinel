"""
Feature engineering for the well-being classification task.
All functions take a raw merged DataFrame and return per-citizen feature DataFrames.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from core.config import STANDARD_EVENT_TYPES, ESCALATED_EVENT_TYPES


def derive_labels(df: pd.DataFrame) -> pd.Series:
    """
    Derive binary labels from the training data.
    Citizens with any escalated event type are label=1.
    This is the ground-truth signal embedded in the training data.
    """
    escalated = df[df["EventType"].isin(ESCALATED_EVENT_TYPES)]["CitizenID"].unique()
    all_citizens = df["CitizenID"].unique()
    return pd.Series(
        {cid: (1 if cid in escalated else 0) for cid in all_citizens},
        name="label",
    )


def compute_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from event types and counts.
    Signal: escalated event types are the strongest indicator of label=1.
    """
    records = []
    for cid, group in df.groupby("CitizenID"):
        n_total = len(group)
        n_routine = (group["EventType"] == "routine check-up").sum()
        n_screening = (group["EventType"] == "preventive screening").sum()
        n_coaching = (group["EventType"] == "lifestyle coaching session").sum()
        n_escalated = group["EventType"].isin(ESCALATED_EVENT_TYPES).sum()
        n_standard = group["EventType"].isin(STANDARD_EVENT_TYPES).sum()

        has_emergency = int((group["EventType"] == "emergency visit").any())
        has_specialist = int((group["EventType"] == "specialist consultation").any())
        has_followup = int((group["EventType"] == "follow-up assessment").any())
        has_any_escalated = int(n_escalated > 0)

        last_event = group.sort_values("Timestamp").iloc[-1]["EventType"]
        last_event_escalated = int(last_event in ESCALATED_EVENT_TYPES)

        escalation_ratio = n_escalated / n_total if n_total > 0 else 0.0

        records.append({
            "CitizenID": cid,
            "n_events": n_total,
            "n_routine": n_routine,
            "n_screening": n_screening,
            "n_coaching": n_coaching,
            "n_escalated": n_escalated,
            "n_standard": n_standard,
            "has_emergency": has_emergency,
            "has_specialist": has_specialist,
            "has_followup": has_followup,
            "has_any_escalated": has_any_escalated,
            "last_event_escalated": last_event_escalated,
            "escalation_ratio": escalation_ratio,
        })
    return pd.DataFrame(records)


def compute_biometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate features from Physical Activity Index, Sleep Quality Index,
    and Environmental Exposure Level.
    Signal: declining PAI/SQI and increasing EEL correlate with label=1.
    """
    records = []
    for cid, group in df.sort_values("Timestamp").groupby("CitizenID"):
        n = len(group)

        for col, prefix in [
            ("PhysicalActivityIndex", "pai"),
            ("SleepQualityIndex", "sqi"),
            ("EnvironmentalExposureLevel", "eel"),
        ]:
            vals = group[col].values.astype(float)
            row: dict = {}

            row[f"{prefix}_mean"] = float(np.mean(vals))
            row[f"{prefix}_std"] = float(np.std(vals)) if n > 1 else 0.0
            row[f"{prefix}_min"] = float(np.min(vals))
            row[f"{prefix}_max"] = float(np.max(vals))
            row[f"{prefix}_first"] = float(vals[0])
            row[f"{prefix}_last"] = float(vals[-1])
            row[f"{prefix}_delta"] = float(vals[-1] - vals[0])

            if n >= 3:
                slope, _, _, _, _ = stats.linregress(range(n), vals)
                row[f"{prefix}_trend"] = float(slope)
            else:
                row[f"{prefix}_trend"] = 0.0

            if n >= 3:
                last_third = vals[n * 2 // 3:]
                first_third = vals[:max(1, n // 3)]
                row[f"{prefix}_late_mean"] = float(np.mean(last_third))
                row[f"{prefix}_early_mean"] = float(np.mean(first_third))
                row[f"{prefix}_late_vs_early"] = float(np.mean(last_third) - np.mean(first_third))
            else:
                row[f"{prefix}_late_mean"] = row[f"{prefix}_mean"]
                row[f"{prefix}_early_mean"] = row[f"{prefix}_mean"]
                row[f"{prefix}_late_vs_early"] = 0.0

            if "CitizenID" not in row:
                row["CitizenID"] = cid
            records.append({**{"CitizenID": cid}, **row})

    # Merge all per-citizen rows (each iteration appends one dict per metric, need to consolidate)
    # Rebuild properly
    proper_records = []
    for cid, group in df.sort_values("Timestamp").groupby("CitizenID"):
        n = len(group)
        row = {"CitizenID": cid}

        for col, prefix in [
            ("PhysicalActivityIndex", "pai"),
            ("SleepQualityIndex", "sqi"),
            ("EnvironmentalExposureLevel", "eel"),
        ]:
            vals = group[col].values.astype(float)
            row[f"{prefix}_mean"] = float(np.mean(vals))
            row[f"{prefix}_std"] = float(np.std(vals)) if n > 1 else 0.0
            row[f"{prefix}_min"] = float(np.min(vals))
            row[f"{prefix}_max"] = float(np.max(vals))
            row[f"{prefix}_first"] = float(vals[0])
            row[f"{prefix}_last"] = float(vals[-1])
            row[f"{prefix}_delta"] = float(vals[-1] - vals[0])

            if n >= 3:
                slope, _, _, _, _ = stats.linregress(range(n), vals)
                row[f"{prefix}_trend"] = float(slope)
            else:
                row[f"{prefix}_trend"] = 0.0

            last_third = vals[max(0, n * 2 // 3):]
            first_third = vals[:max(1, n // 3)]
            row[f"{prefix}_late_vs_early"] = float(np.mean(last_third) - np.mean(first_third))

        proper_records.append(row)

    return pd.DataFrame(proper_records)


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features from event timing patterns.
    """
    records = []
    for cid, group in df.sort_values("Timestamp").groupby("CitizenID"):
        ts = pd.to_datetime(group["Timestamp"])
        n = len(group)

        span_days = (ts.max() - ts.min()).days if n > 1 else 0
        events_per_month = (n / (span_days / 30.0)) if span_days > 0 else n

        if n > 1:
            gaps = ts.diff().dropna().dt.days.values
            gap_mean = float(np.mean(gaps))
            gap_std = float(np.std(gaps))
        else:
            gap_mean = 0.0
            gap_std = 0.0

        records.append({
            "CitizenID": cid,
            "span_days": float(span_days),
            "events_per_month": float(events_per_month),
            "inter_event_gap_mean": gap_mean,
            "inter_event_gap_std": gap_std,
        })
    return pd.DataFrame(records)


def compute_persona_features(personas: dict) -> pd.DataFrame:
    """
    Extract numeric features from parsed persona profiles.
    Signals: age, mobility level, health risk indicators, social isolation.
    """
    def mobility_score(text: str) -> float:
        t = text.lower()
        if "very low" in t:
            return 0.0
        if "low-moderate" in t or "low moderate" in t:
            return 1.5
        if "low" in t:
            return 1.0
        if "variable" in t:
            return 2.5
        if "moderate" in t:
            return 2.0
        if "high" in t:
            return 3.0
        return 1.5

    def health_risk_score(text: str) -> float:
        t = text.lower()
        risk = ["irregular", "avoids", "unstable", "declining", "poor", "heavy reliance",
                "sleep issues", "back pain", "cancelling", "run-down", "persistent"]
        protective = ["stable", "well-managed", "active", "coordinated", "regular", "gym"]
        score = sum(1.0 for kw in risk if kw in t)
        score -= sum(0.5 for kw in protective if kw in t)
        return max(0.0, score)

    def social_isolation_score(text: str) -> float:
        t = text.lower()
        isolated = ["cancelling", "alone", "narrowed", "small", "rare", "increasingly",
                    "spending more time alone", "avoids"]
        connected = ["broad", "daily", "regular", "active", "frequent", "wide"]
        score = sum(1.0 for kw in isolated if kw in t)
        score -= sum(0.5 for kw in connected if kw in t)
        return max(0.0, score)

    records = []
    for cid, p in personas.items():
        age = p.get("age") or 0
        occ = (p.get("occupation") or "").lower()
        mob = p.get("mobility") or ""
        health = p.get("health_behavior") or ""
        social = p.get("social_pattern") or ""
        full = p.get("full_text") or ""

        records.append({
            "CitizenID": cid,
            "persona_age": float(age),
            "persona_age_over_75": int(age >= 75),
            "persona_is_retired": int("retired" in occ),
            "persona_mobility_score": mobility_score(mob),
            "persona_health_risk_score": health_risk_score(health + " " + full),
            "persona_social_isolation_score": social_isolation_score(social + " " + full),
        })
    return pd.DataFrame(records)


def compute_user_features(df: pd.DataFrame, users_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Features from user profile (age, job category).
    """
    if users_df is None or users_df.empty:
        return df[["CitizenID"]].drop_duplicates()

    records = []
    for _, user in users_df.iterrows():
        age = 2026 - int(user.get("birth_year", 2000))
        job = str(user.get("job", "unknown")).lower()
        job_is_physical = int(any(k in job for k in ["driver", "worker", "farmer", "builder"]))
        records.append({
            "CitizenID": user["user_id"],
            "age": age,
            "job_is_physical": job_is_physical,
        })
    return pd.DataFrame(records)


def build_feature_matrix(
    status_df: pd.DataFrame,
    users_df: pd.DataFrame | None,
    feature_groups: list[str],
    personas: dict | None = None,
) -> pd.DataFrame:
    """
    Build a citizen-level feature matrix from the requested feature groups.
    Returns a DataFrame with CitizenID as first column.
    """
    citizens = status_df[["CitizenID"]].drop_duplicates().reset_index(drop=True)
    dfs = [citizens]

    if "event_features" in feature_groups:
        dfs.append(compute_event_features(status_df))

    if "biometric_features" in feature_groups:
        dfs.append(compute_biometric_features(status_df))

    if "temporal_features" in feature_groups:
        dfs.append(compute_temporal_features(status_df))

    if "user_features" in feature_groups:
        dfs.append(compute_user_features(status_df, users_df))

    if "persona_features" in feature_groups and personas:
        dfs.append(compute_persona_features(personas))

    result = citizens.copy()
    for feat_df in dfs[1:]:
        result = result.merge(feat_df, on="CitizenID", how="left")

    result = result.fillna(result.median(numeric_only=True))
    return result
