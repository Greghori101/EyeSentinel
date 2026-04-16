from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")
CHALLENGE_DATA_DIR = BASE_DIR / "challenge_data"
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openrouter/elephant-alpha")


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


@dataclass(frozen=True)
class ScenarioConfig:
    split: str
    scenario_name: str
    dataset_dir: Path

    @property
    def slug(self) -> str:
        return slugify(self.scenario_name)

    @property
    def family_name(self) -> str:
        return scenario_family_name(self.scenario_name)

    @property
    def family_slug(self) -> str:
        return slugify(self.family_name)

    @property
    def output_path(self) -> Path:
        return OUTPUT_DIR / self.split / f"{self.slug}_predictions.txt"


def discover_scenarios(
    split: str | None = None, scenario: str | None = None
) -> list[ScenarioConfig]:
    requested_splits = [split] if split else ["training", "evaluation"]
    configs: list[ScenarioConfig] = []

    for split_name in requested_splits:
        split_dir = CHALLENGE_DATA_DIR / split_name
        if not split_dir.exists():
            continue

        for dataset_dir in sorted(
            path for path in split_dir.iterdir() if path.is_dir()
        ):
            if (
                scenario
                and dataset_dir.name != scenario
                and slugify(dataset_dir.name) != slugify(scenario)
            ):
                continue
            configs.append(
                ScenarioConfig(
                    split=split_name,
                    scenario_name=dataset_dir.name,
                    dataset_dir=dataset_dir,
                )
            )

    if not configs:
        raise FileNotFoundError(
            f"No datasets found for split={split!r} scenario={scenario!r} under {CHALLENGE_DATA_DIR}"
        )
    return configs


def scenario_family_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s*-\s*(train|validation)\s*$", "", value, flags=re.IGNORECASE)
    return value.strip()


def new_session_id() -> str:
    return f"reply-mirror-{uuid.uuid4().hex[:16]}"
