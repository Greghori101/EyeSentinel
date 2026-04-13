"""
Configuration for each level of the challenge.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

STANDARD_EVENT_TYPES = {
    "routine check-up",
    "preventive screening",
    "lifestyle coaching session",
}

ESCALATED_EVENT_TYPES = {
    "follow-up assessment",
    "specialist consultation",
    "emergency visit",
    "urgent care visit",
    "hospital admission",
}


@dataclass
class LevelConfig:
    level: int
    train_dir: Path
    eval_dir: Path
    llm_model: str = "meta-llama/llama-3.1-8b-instruct:free"

    @property
    def train_status(self) -> Path:
        return self.train_dir / "status.csv"

    @property
    def train_locations(self) -> Path:
        return self.train_dir / "locations.json"

    @property
    def train_users(self) -> Path:
        return self.train_dir / "users.json"

    @property
    def train_personas(self) -> Path:
        return self.train_dir / "personas.md"

    @property
    def eval_status(self) -> Path:
        return self.eval_dir / "status.csv"

    @property
    def eval_locations(self) -> Path:
        return self.eval_dir / "locations.json"

    @property
    def eval_users(self) -> Path:
        return self.eval_dir / "users.json"

    @property
    def eval_personas(self) -> Path:
        return self.eval_dir / "personas.md"


def get_level_config(level: int, model: str | None = None) -> LevelConfig:
    train_dir = DATA_DIR / "training" / f"public_lev_{level}"
    eval_dir = DATA_DIR / "evaluation" / f"public_lev_{level}"
    default_model = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    return LevelConfig(
        level=level,
        train_dir=train_dir,
        eval_dir=eval_dir,
        llm_model=model or default_model,
    )
