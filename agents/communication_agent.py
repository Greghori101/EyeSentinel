from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from agents.data_agent import normalize_name, normalize_text
from core.feature_store import DatasetBundle, FeatureStore


PHISHING_KEYWORDS = [
    "urgent",
    "verify",
    "suspended",
    "locked",
    "security",
    "customs",
    "fee",
    "pay now",
    "failed",
    "suspicious",
    "login",
    "identity",
    "restore access",
    "benefit",
    "billing",
    "renewal",
    "password",
    "secure",
    "account",
    "review required",
    "confirm",
]
SHORT_LINK_HINTS = ["bit.ly", "tinyurl", "goo.gl"]
LOOKALIKE_HINTS = [
    "paypa1",
    "amaz0n",
    "netfl1x",
    "secure-pay",
    "verify2087",
    "ssa-secure",
]
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
DATE_RE = re.compile(r"^Date:\s*(.+)$", re.MULTILINE)
TO_RE = re.compile(r'^To:\s*"?([^"<\n]+)"?', re.MULTILINE)


class CommunicationAgent:
    def __init__(self, store: FeatureStore):
        self.store = store

    def run(self, bundle: DatasetBundle) -> pd.DataFrame:
        actor_ids_by_name = self._build_actor_name_maps(bundle.actor_directory)
        messages = self._parse_messages(bundle.sms, bundle.mails, actor_ids_by_name)
        audio_events = self._attach_audio_actor_ids(
            bundle.audio_events, actor_ids_by_name
        )

        comm_summary = self._build_actor_communication_summary(
            messages, audio_events, bundle.actor_directory
        )
        features = self._build_transaction_communication_features(
            bundle.transactions, comm_summary
        )
        self.store.metadata["parsed_messages"] = messages
        self.store.set_features("communication", features)
        return features

    def _build_actor_name_maps(
        self, actor_directory: pd.DataFrame
    ) -> dict[str, dict[str, list[str]]]:
        full_name_map: defaultdict[str, list[str]] = defaultdict(list)
        first_name_map: defaultdict[str, list[str]] = defaultdict(list)
        for row in actor_directory.itertuples(index=False):
            actor_id = row.actor_id
            full_name = normalize_name(getattr(row, "full_name", ""))
            first_name = normalize_name(getattr(row, "first_name_norm", ""))
            if full_name:
                full_name_map[full_name].append(actor_id)
            if first_name:
                first_name_map[first_name].append(actor_id)
        return {"full": dict(full_name_map), "first": dict(first_name_map)}

    def _parse_messages(
        self,
        sms: list[dict],
        mails: list[dict],
        actor_ids_by_name: dict[str, dict[str, list[str]]],
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for payload in sms:
            text = payload.get("sms") or payload.get("message") or payload.get("value") or ""
            rows.append(self._message_row(text, "sms", actor_ids_by_name, payload))
        for payload in mails:
            text = payload.get("mail") or payload.get("message") or payload.get("value") or ""
            rows.append(self._message_row(text, "mail", actor_ids_by_name, payload))
        return pd.DataFrame(rows)

    def _message_row(
        self,
        text: str,
        source: str,
        actor_ids_by_name: dict[str, dict[str, list[str]]],
        payload: dict,
    ) -> dict:
        normalized_text = normalize_text(text)
        actor_ids = self._match_actor_ids(text, normalized_text, actor_ids_by_name, payload)
        date_match = DATE_RE.search(text)
        timestamp = (
            self._coerce_timestamp(date_match.group(1)) if date_match else pd.NaT
        )
        return {
            "source": source,
            "timestamp": timestamp,
            "actor_ids": actor_ids,
            "phishing_score": self._phishing_score(text),
            "contains_short_link": float(
                any(hint in text.lower() for hint in SHORT_LINK_HINTS)
            ),
            "contains_iban": float(bool(IBAN_RE.search(text))),
            "text": text,
        }

    def _match_actor_ids(
        self,
        text: str,
        normalized_text: str,
        actor_ids_by_name: dict[str, dict[str, list[str]]],
        payload: dict,
    ) -> list[str]:
        matched: set[str] = set()
        explicit_user_id = payload.get("user_id") or payload.get("user") or payload.get("biotag")
        if explicit_user_id:
            matched.add(str(explicit_user_id))
        to_match = TO_RE.search(text)
        if to_match:
            name = normalize_name(to_match.group(1))
            matched.update(actor_ids_by_name["full"].get(name, []))

        for name, actor_ids in actor_ids_by_name["full"].items():
            if name and name in normalized_text:
                matched.update(actor_ids)

        for first_name, actor_ids in actor_ids_by_name["first"].items():
            if (
                len(actor_ids) == 1
                and first_name
                and re.search(rf"\b{re.escape(first_name)}\b", normalized_text)
            ):
                matched.update(actor_ids)
        return sorted(matched)

    def _attach_audio_actor_ids(
        self,
        audio_events: pd.DataFrame,
        actor_ids_by_name: dict[str, dict[str, list[str]]],
    ) -> pd.DataFrame:
        if audio_events.empty:
            return pd.DataFrame(columns=["timestamp", "actor_ids"])
        events = audio_events.copy()
        events["actor_ids"] = events["speaker_name_norm"].map(
            lambda name: actor_ids_by_name["full"].get(name, [])
        )
        return events

    def _build_actor_communication_summary(
        self,
        messages: pd.DataFrame,
        audio_events: pd.DataFrame,
        actor_directory: pd.DataFrame,
    ) -> dict[str, dict[str, object]]:
        user_vulnerability = self._user_vulnerability(actor_directory)
        actor_message_history: defaultdict[str, list[dict]] = defaultdict(list)
        for row in messages.itertuples(index=False):
            for actor_id in row.actor_ids:
                actor_message_history[actor_id].append(
                    {
                        "timestamp": row.timestamp,
                        "phishing_score": float(row.phishing_score),
                        "contains_short_link": float(row.contains_short_link),
                    }
                )

        actor_audio_history: defaultdict[str, list[pd.Timestamp]] = defaultdict(list)
        for row in audio_events.itertuples(index=False):
            for actor_id in row.actor_ids:
                actor_audio_history[actor_id].append(row.timestamp)

        summary: dict[str, dict[str, object]] = {}
        actor_ids = (
            set(actor_directory["actor_id"].tolist())
            if not actor_directory.empty
            else set()
        )
        for actor_id in actor_ids:
            summary[actor_id] = {
                "messages": sorted(
                    actor_message_history.get(actor_id, []),
                    key=lambda item: (
                        item["timestamp"]
                        if pd.notna(item["timestamp"])
                        else pd.Timestamp.min
                    ),
                ),
                "audio": sorted(
                    ts for ts in actor_audio_history.get(actor_id, []) if pd.notna(ts)
                ),
                "vulnerability": float(user_vulnerability.get(actor_id, 0.0)),
            }
        return summary

    def _build_transaction_communication_features(
        self,
        transactions: pd.DataFrame,
        comm_summary: dict[str, dict[str, object]],
    ) -> pd.DataFrame:
        records: list[dict] = []
        tx = transactions.copy().sort_values("timestamp")
        for row in tx.itertuples(index=False):
            sender = self._recent_comm_features(
                comm_summary.get(row.sender_id, {}), row.timestamp
            )
            recipient = self._recent_comm_features(
                comm_summary.get(row.recipient_id, {}), row.timestamp
            )
            records.append(
                {
                    "transaction_id": row.transaction_id,
                    "sender_recent_phishing_7d": sender["phishing_7d"],
                    "sender_recent_phishing_30d": sender["phishing_30d"],
                    "sender_recent_short_links": sender["short_links_30d"],
                    "sender_audio_30d": sender["audio_30d"],
                    "sender_vulnerability": sender["vulnerability"],
                    "recipient_recent_phishing_7d": recipient["phishing_7d"],
                    "recipient_recent_phishing_30d": recipient["phishing_30d"],
                    "recipient_audio_30d": recipient["audio_30d"],
                    "recipient_vulnerability": recipient["vulnerability"],
                }
            )
        return pd.DataFrame(records).fillna(0.0)

    @staticmethod
    def _coerce_timestamp(value: str) -> pd.Timestamp:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return pd.NaT
        return ts.tz_localize(None)

    def _recent_comm_features(
        self, summary: dict[str, object], tx_time: pd.Timestamp
    ) -> dict[str, float]:
        messages = summary.get("messages", []) if summary else []
        audio = summary.get("audio", []) if summary else []

        phishing_7d = 0.0
        phishing_30d = 0.0
        short_links_30d = 0.0
        for message in messages:
            ts = message["timestamp"]
            if pd.isna(ts) or ts > tx_time:
                continue
            delta_days = (tx_time - ts).total_seconds() / 86400.0
            if delta_days <= 30:
                phishing_30d = max(phishing_30d, float(message["phishing_score"]))
                short_links_30d += float(message["contains_short_link"])
            if delta_days <= 7:
                phishing_7d = max(phishing_7d, float(message["phishing_score"]))

        audio_30d = 0.0
        for ts in audio:
            if ts <= tx_time and (tx_time - ts).total_seconds() <= 30 * 86400:
                audio_30d += 1.0

        return {
            "phishing_7d": phishing_7d,
            "phishing_30d": phishing_30d,
            "short_links_30d": short_links_30d,
            "audio_30d": audio_30d,
            "vulnerability": (
                float(summary.get("vulnerability", 0.0)) if summary else 0.0
            ),
        }

    def _user_vulnerability(self, actor_directory: pd.DataFrame) -> dict[str, float]:
        vulnerability: dict[str, float] = {}
        if actor_directory.empty:
            return vulnerability

        keywords = [
            "phishing",
            "trust",
            "vulnerable",
            "not immune",
            "careful",
            "suspicious",
            "trap",
        ]
        probability_re = re.compile(r"(\d{1,3})\s*%")
        for row in actor_directory.itertuples(index=False):
            text = normalize_text(getattr(row, "description", "") or "")
            score = 0.0
            probability_match = probability_re.search(text)
            if probability_match:
                score += min(float(probability_match.group(1)) / 100.0, 1.0)
            score += min(sum(1 for kw in keywords if kw in text) / 4.0, 1.0)
            age = float(getattr(row, "age", 0.0) or 0.0)
            if age >= 70:
                score += 0.15
            vulnerability[row.actor_id] = float(min(score, 1.0))
        return vulnerability

    @staticmethod
    def _phishing_score(text: str) -> float:
        lowered = text.lower()
        score = 0.0
        score += min(
            sum(1 for keyword in PHISHING_KEYWORDS if keyword in lowered) * 0.12, 0.72
        )
        if any(hint in lowered for hint in SHORT_LINK_HINTS):
            score += 0.2
        if any(hint in lowered for hint in LOOKALIKE_HINTS):
            score += 0.3
        if "http://" in lowered:
            score += 0.2
        if re.search(r"https?://[^\s]+", lowered):
            score += 0.1
        return float(min(score, 1.0))
