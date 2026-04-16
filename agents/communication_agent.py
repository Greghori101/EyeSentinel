from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agents.data_agent import normalize_name, normalize_text
from core.config import BASE_DIR, DEFAULT_MODEL
from core.feature_store import DatasetBundle, FeatureStore
from core.llm_client import LLMClient


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
CACHE_VERSION = "v2"


class CommunicationAgent:
    def __init__(self, store: FeatureStore):
        self.store = store
        self.cache_path = (
            BASE_DIR / "outputs" / "cache" / "communication_llm_cache.json"
        )
        self.audio_cache_path = (
            BASE_DIR / "outputs" / "cache" / "audio_transcripts.json"
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._llm_cache = self._load_cache()
        self._audio_cache = self._load_cache(self.audio_cache_path)
        self._llm_client: LLMClient | None = None
        self._whisper_model = None

    def run(
        self,
        bundle: DatasetBundle,
        reference_profile: dict | None = None,
    ) -> pd.DataFrame:
        actor_ids_by_name = self._build_actor_name_maps(bundle.actor_directory)
        messages = self._parse_messages(bundle.sms, bundle.mails, actor_ids_by_name)
        messages = self._apply_llm_scores(messages)
        messages = self._apply_reference_memory(messages, reference_profile)
        audio_events = self._attach_audio_actor_ids(
            bundle.audio_events, actor_ids_by_name
        )
        audio_events = self._apply_audio_analysis(audio_events)

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
            text = (
                payload.get("sms")
                or payload.get("message")
                or payload.get("value")
                or ""
            )
            rows.append(self._message_row(text, "sms", actor_ids_by_name, payload))
        for payload in mails:
            text = (
                payload.get("mail")
                or payload.get("message")
                or payload.get("value")
                or ""
            )
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
        actor_ids = self._match_actor_ids(
            text, normalized_text, actor_ids_by_name, payload
        )
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
            "normalized_text": normalized_text,
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
        explicit_user_id = (
            payload.get("user_id") or payload.get("user") or payload.get("biotag")
        )
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
            return pd.DataFrame(columns=["timestamp", "actor_ids", "audio_risk"])
        events = audio_events.copy()
        events["actor_ids"] = events["speaker_name_norm"].map(
            lambda name: actor_ids_by_name["full"].get(name, [])
        )
        events["audio_risk"] = 0.0
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
                        "llm_score": float(getattr(row, "llm_fraud_score", 0.0)),
                        "memory_score": float(getattr(row, "memory_score", 0.0)),
                        "message_risk": float(
                            getattr(row, "message_risk", row.phishing_score)
                        ),
                        "contains_short_link": float(row.contains_short_link),
                    }
                )

        actor_audio_history: defaultdict[str, list[pd.Timestamp]] = defaultdict(list)
        for row in audio_events.itertuples(index=False):
            for actor_id in row.actor_ids:
                actor_audio_history[actor_id].append(row.timestamp)
                actor_message_history[actor_id].append(
                    {
                        "timestamp": row.timestamp,
                        "phishing_score": float(getattr(row, "audio_risk", 0.0)),
                        "llm_score": float(getattr(row, "audio_risk", 0.0)),
                        "memory_score": 0.0,
                        "message_risk": float(getattr(row, "audio_risk", 0.0)),
                        "contains_short_link": 0.0,
                    }
                )

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
                    "sender_recent_llm_7d": sender["llm_7d"],
                    "sender_recent_llm_30d": sender["llm_30d"],
                    "sender_recent_memory_30d": sender["memory_30d"],
                    "sender_recent_message_risk_30d": sender["message_risk_30d"],
                    "sender_recent_short_links": sender["short_links_30d"],
                    "sender_audio_30d": sender["audio_30d"],
                    "sender_vulnerability": sender["vulnerability"],
                    "recipient_recent_phishing_7d": recipient["phishing_7d"],
                    "recipient_recent_phishing_30d": recipient["phishing_30d"],
                    "recipient_recent_llm_30d": recipient["llm_30d"],
                    "recipient_recent_memory_30d": recipient["memory_30d"],
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
        llm_7d = 0.0
        llm_30d = 0.0
        memory_30d = 0.0
        message_risk_30d = 0.0
        short_links_30d = 0.0
        for message in messages:
            ts = message["timestamp"]
            if pd.isna(ts) or ts > tx_time:
                continue
            delta_days = (tx_time - ts).total_seconds() / 86400.0
            if delta_days <= 30:
                phishing_30d = max(phishing_30d, float(message["phishing_score"]))
                llm_30d = max(llm_30d, float(message.get("llm_score", 0.0)))
                memory_30d = max(memory_30d, float(message.get("memory_score", 0.0)))
                message_risk_30d = max(
                    message_risk_30d, float(message.get("message_risk", 0.0))
                )
                short_links_30d += float(message["contains_short_link"])
            if delta_days <= 7:
                phishing_7d = max(phishing_7d, float(message["phishing_score"]))
                llm_7d = max(llm_7d, float(message.get("llm_score", 0.0)))

        audio_30d = 0.0
        for ts in audio:
            if ts <= tx_time and (tx_time - ts).total_seconds() <= 30 * 86400:
                audio_30d += 1.0

        return {
            "phishing_7d": phishing_7d,
            "phishing_30d": phishing_30d,
            "llm_7d": llm_7d,
            "llm_30d": llm_30d,
            "memory_30d": memory_30d,
            "message_risk_30d": message_risk_30d,
            "short_links_30d": short_links_30d,
            "audio_30d": audio_30d,
            "vulnerability": (
                float(summary.get("vulnerability", 0.0)) if summary else 0.0
            ),
        }

    def _apply_llm_scores(self, messages: pd.DataFrame) -> pd.DataFrame:
        if messages.empty:
            messages["llm_fraud_score"] = []
            messages["message_risk"] = []
            return messages

        frame = messages.copy()
        frame["llm_fraud_score"] = 0.0
        llm_enabled = os.getenv("ENABLE_COMM_LLM", "true").lower() not in {
            "0",
            "false",
            "no",
        }
        if llm_enabled and self._can_use_llm():
            budget = int(os.getenv("LLM_MESSAGE_LIMIT", "40"))
            candidate_rows = frame[
                (frame["phishing_score"] >= 0.18)
                | (frame["contains_short_link"] >= 1.0)
                | (frame["contains_iban"] >= 1.0)
                | (frame["source"] == "mail")
            ].copy()
            candidate_rows["cache_key"] = candidate_rows["normalized_text"].map(
                self._cache_key
            )
            unique_candidates = (
                candidate_rows.drop_duplicates(subset=["cache_key"])
                .sort_values(
                    ["phishing_score", "contains_short_link", "contains_iban"],
                    ascending=False,
                )
                .head(budget)
            )
            for row in unique_candidates.itertuples(index=False):
                cached = self._llm_cache.get(row.cache_key)
                if cached is None or self._needs_refresh(cached):
                    cached = self._analyze_text_with_llm(row.text, row.source)
                    self._llm_cache[row.cache_key] = cached
                frame.loc[
                    frame["normalized_text"] == row.normalized_text, "llm_fraud_score"
                ] = float(cached.get("fraud_score", 0.0))
            self._save_cache()

        frame["message_risk"] = (
            0.45 * frame["phishing_score"].astype(float)
            + 0.45 * frame["llm_fraud_score"].astype(float)
            + 0.05 * frame["contains_short_link"].astype(float)
            + 0.05 * frame["contains_iban"].astype(float)
        ).clip(0.0, 1.0)
        return frame

    def _apply_reference_memory(
        self,
        messages: pd.DataFrame,
        reference_profile: dict | None,
    ) -> pd.DataFrame:
        if messages.empty:
            messages["memory_score"] = []
            return messages
        frame = messages.copy()
        frame["memory_score"] = 0.0
        memory_items = (reference_profile or {}).get("message_memory", [])
        reference_texts = [item["text"] for item in memory_items if item.get("text")]
        if not reference_texts:
            frame["message_risk"] = frame.get("message_risk", frame["phishing_score"])
            return frame

        current_texts = frame["normalized_text"].fillna("").tolist()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4096)
        all_vectors = vectorizer.fit_transform(reference_texts + current_texts)
        ref_vectors = all_vectors[: len(reference_texts)]
        cur_vectors = all_vectors[len(reference_texts) :]
        sims = cosine_similarity(cur_vectors, ref_vectors)
        ref_scores = pd.Series(
            [float(item.get("score", 0.0)) for item in memory_items], dtype=float
        ).to_numpy()
        weighted = sims * ref_scores
        frame["memory_score"] = weighted.max(axis=1) if weighted.size else 0.0
        frame["message_risk"] = frame[["message_risk", "memory_score"]].max(axis=1)
        return frame

    def _can_use_llm(self) -> bool:
        try:
            self._get_llm_client()
            return True
        except Exception:
            return False

    def _get_llm_client(self) -> LLMClient:
        if self._llm_client is None:
            session_id = str(
                self.store.metadata.get("session_id", "reply-mirror-local")
            )
            self._llm_client = LLMClient(model=DEFAULT_MODEL, session_id=session_id)
        return self._llm_client

    def _analyze_text_with_llm(self, text: str, source: str) -> dict[str, Any]:
        prompt = (
            "You are a fraud-intelligence agent for MirrorPay. "
            "Analyze the following communication for social-engineering, credential theft, payment redirection, "
            "fake urgency, or fraud coordination. "
            "Return strict JSON with keys fraud_score (0..1), urgency_score (0..1), and rationale.\n\n"
            f"Source: {source}\n"
            f"Communication:\n{text[:6000]}"
        )
        try:
            response = self._get_llm_client().chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            parsed = self._parse_llm_json(response.get("content", ""))
            fraud_score = float(parsed.get("fraud_score", 0.0))
            urgency_score = float(parsed.get("urgency_score", 0.0))
            return {
                "fraud_score": max(0.0, min(1.0, fraud_score)),
                "urgency_score": max(0.0, min(1.0, urgency_score)),
                "rationale": str(parsed.get("rationale", ""))[:500],
                "cache_version": CACHE_VERSION,
            }
        except Exception:
            return {
                "fraud_score": 0.0,
                "urgency_score": 0.0,
                "rationale": "",
                "cache_version": CACHE_VERSION,
            }

    @staticmethod
    def _parse_llm_json(content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            return {}
        try:
            return json.loads(content)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return {}
            return {}

    def _load_cache(self, path: Path | None = None) -> dict[str, Any]:
        target = path or self.cache_path
        if not target.exists():
            return {}
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self) -> None:
        self._save_cache_to(self.cache_path, self._llm_cache)

    def _save_cache_to(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _apply_audio_analysis(self, audio_events: pd.DataFrame) -> pd.DataFrame:
        if audio_events.empty:
            return audio_events
        if os.getenv("ENABLE_AUDIO_AGENT", "true").lower() in {"0", "false", "no"}:
            return audio_events
        if importlib.util.find_spec("whisper") is None:
            return audio_events

        events = audio_events.copy()
        for row in events.itertuples(index=False):
            file_path = str(getattr(row, "file_path", "") or "")
            if not file_path:
                continue
            key = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
            cached = self._audio_cache.get(key)
            if cached is None:
                transcript = self._transcribe_audio(file_path)
                llm_result = (
                    self._analyze_text_with_llm(transcript, "audio")
                    if transcript
                    else {}
                )
                cached = {
                    "transcript": transcript[:2000],
                    "audio_risk": float(llm_result.get("fraud_score", 0.0)),
                }
                self._audio_cache[key] = cached
            events.loc[events["file_path"] == file_path, "audio_risk"] = float(
                cached.get("audio_risk", 0.0)
            )

        self._save_cache_to(self.audio_cache_path, self._audio_cache)
        return events

    def _transcribe_audio(self, file_path: str) -> str:
        try:
            if self._whisper_model is None:
                import whisper  # type: ignore

                self._whisper_model = whisper.load_model(
                    os.getenv("WHISPER_MODEL", "base")
                )
            result = self._whisper_model.transcribe(file_path)
            return str(result.get("text", "")).strip()
        except Exception:
            return ""

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _needs_refresh(cached: dict[str, Any]) -> bool:
        return cached.get("cache_version") != CACHE_VERSION

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
