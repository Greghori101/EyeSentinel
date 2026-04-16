from __future__ import annotations
import argparse

from core.config import OUTPUT_DIR, discover_scenarios, new_session_id
from core.feature_store import FeatureStore
from agents.orchestrator import FraudOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Reply Mirror fraud pipeline on challenge_data."
    )
    parser.add_argument(
        "--split",
        choices=["training", "evaluation"],
        default="evaluation",
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario folder name or slug.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all splits instead of only the selected split.",
    )
    return parser.parse_args()


def run_pipeline(
    split: str = "evaluation", scenario: str | None = None, run_all: bool = False
) -> list[dict]:
    session_id = new_session_id()
    print("Reply Mirror Fraud Pipeline")
    print(f"Session ID: {session_id}\n")

    selected_split = None if run_all else split
    configs = discover_scenarios(split=selected_split, scenario=scenario)
    results: list[dict] = []

    for config in configs:
        split_label = "train" if config.split == "training" else "eval"
        print(f"[{split_label}] dataset={config.scenario_name}")
        print(f"  session_id:   {session_id}")
        store = FeatureStore()
        orchestrator = FraudOrchestrator(store=store, output_dir=OUTPUT_DIR)
        result = orchestrator.run(config)
        session_txt_path = _write_session_id_files(
            config.split, config.slug, session_id
        )
        result["session_id"] = session_id
        result["session_txt_path"] = str(session_txt_path)
        results.append(result)
        print(f"  transactions: {result['n_transactions']}")
        print(f"  flagged:      {result['n_flagged_transactions']}")
        print(f"  output_txt:   {result['output_path']}")
        print(f"  session_txt:  {result['session_txt_path']}")
        print(f"  threshold:    {result['threshold']:.6f}\n")

    return results


def _write_session_id_files(split: str, slug: str, session_id: str) -> str:
    split_dir = OUTPUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    per_dataset_path = split_dir / f"{slug}_session_id.txt"
    per_dataset_path.write_text(session_id + "\n", encoding="utf-8")

    manifest_path = OUTPUT_DIR / "session_ids.txt"
    entries: dict[str, str] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                entries[key.strip()] = value.strip()

    entries[f"{split}/{slug}"] = session_id

    lines: list[str] = []
    for key in sorted(entries):
        lines.append(f"{key}: {entries[key]}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return str(per_dataset_path)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(split=args.split, scenario=args.scenario, run_all=args.all)
