"""
Main pipeline entry point.

Usage:
    python pipeline.py [level]          # level: 1, 2, or 3 (default: 1)

Output:
    - outputs/level{N}_predictions.txt  (submit this)
    - Prints the Langfuse Session ID    (submit this alongside the output file)

The submission requires:
    1. Langfuse Session ID (printed at end of run)
    2. Output .txt file (outputs/level{N}_predictions.txt)
    3. Source code .zip (for evaluation datasets only)
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from core.config import get_level_config
from core.feature_store import FeatureStore
from core.langfuse_setup import new_session_id, configure
from core.llm_client import LLMClient
from agents.data_agent import DataIngestionAgent
from agents.feature_agent import FeatureEngineeringAgent
from agents.analysis_agent import PatternAnalysisAgent
from agents.training_agent import ModelTrainingAgent
from agents.threshold_agent import ThresholdTuningAgent
from agents.orchestrator import OrchestratorAgent
from langfuse.decorators import observe


BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@observe(name="full_pipeline")
def run_pipeline(level: int = 1, model: str | None = None) -> tuple[str, str]:
    """
    Run the complete classification pipeline for the given level.

    Returns:
        (session_id, output_path) — both needed for submission.
    """
    session_id = new_session_id()
    configure(session_id)

    print(f"\n{'='*60}")
    print(f"Reply Mirror Challenge — Level {level}")
    print(f"Session ID: {session_id}")
    print(f"{'='*60}\n")

    config = get_level_config(level, model)
    store = FeatureStore()

    llm = LLMClient(model=config.llm_model, session_id=session_id)

    agents = {
        "data": DataIngestionAgent(config, store),
        "feature": FeatureEngineeringAgent(config, store),
        "analysis": PatternAnalysisAgent(config, store),
        "training": ModelTrainingAgent(config, store),
        "threshold": ThresholdTuningAgent(config, store),
    }

    orchestrator = OrchestratorAgent(
        config=config,
        store=store,
        llm=llm,
        agents=agents,
        output_dir=OUTPUT_DIR,
    )

    orchestrator.run()

    output_path = str(OUTPUT_DIR / f"level{level}_predictions.txt")

    print(f"\n{'='*60}")
    print(f"DONE — Level {level}")
    print(f"Output file: {output_path}")
    print(f"\n>>> LANGFUSE SESSION ID: {session_id} <<<")
    print(f"Submit this Session ID alongside your output file.")
    print(f"{'='*60}\n")

    return session_id, output_path


if __name__ == "__main__":
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model = sys.argv[2] if len(sys.argv) > 2 else None
    run_pipeline(level, model)
