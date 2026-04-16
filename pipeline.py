from __future__ import annotations
import argparse
import os

from dotenv import load_dotenv
from langfuse import Langfuse, get_client

load_dotenv()

from core.config import OUTPUT_DIR, ScenarioConfig, discover_scenarios
from core.feature_store import FeatureStore
from core.langfuse_setup import configure, new_session_id
from agents.orchestrator import FraudOrchestrator


langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


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
    print("Reply Mirror Fraud Pipeline")
    print("Session IDs are generated per dataset.\n")

    training_refs: dict[str, dict] = {}
    results: list[dict] = []

    if run_all:
        training_configs = discover_scenarios(split="training", scenario=scenario)
        evaluation_configs = discover_scenarios(split="evaluation", scenario=scenario)

        for config in training_configs:
            result, reference_profile = _run_single_config(config)
            training_refs[config.family_slug] = reference_profile
            results.append(result)

        for config in evaluation_configs:
            result, _ = _run_single_config(
                config,
                reference_profile=training_refs.get(config.family_slug),
            )
            results.append(result)
        return results

    configs = discover_scenarios(split=split, scenario=scenario)
    training_lookup = (
        {config.family_slug: config for config in discover_scenarios(split="training")}
        if split == "evaluation"
        else {}
    )

    for config in configs:
        reference_profile = None
        if config.split == "evaluation":
            paired_training = training_lookup.get(config.family_slug)
            if paired_training is not None:
                _, reference_profile = _run_single_config(
                    paired_training,
                    write_output=False,
                    write_session=False,
                    announce=False,
                )
        result, _ = _run_single_config(config, reference_profile=reference_profile)
        results.append(result)

    return results


def _run_single_config(
    config: ScenarioConfig,
    reference_profile: dict | None = None,
    write_output: bool = True,
    write_session: bool = True,
    announce: bool = True,
) -> tuple[dict, dict | None]:
    session_id = new_session_id()
    configure(session_id)
    split_label = "train" if config.split == "training" else "eval"
    if announce:
        print(f"[{split_label}] dataset={config.scenario_name}")
        print(f"  session_id:   {session_id}")

    store = FeatureStore()
    store.metadata["session_id"] = session_id
    orchestrator = FraudOrchestrator(store=store, output_dir=OUTPUT_DIR)
    result = orchestrator.run(
        config,
        reference_profile=reference_profile,
        write_output=write_output,
    )
    session_manifest_path = ""
    if write_session:
        session_manifest_path = str(
            _write_session_manifest(config.split, config.slug, session_id)
        )
    result["session_id"] = session_id
    result["session_manifest_path"] = session_manifest_path

    if announce:
        print(f"  transactions: {result['n_transactions']}")
        print(f"  flagged:      {result['n_flagged_transactions']}")
        if result["output_path"]:
            print(f"  output_txt:   {result['output_path']}")
        if result["session_manifest_path"]:
            print(f"  session_txt:  {result['session_manifest_path']}")
        if reference_profile is not None:
            print("  reference:    paired training profile")
        print(f"  threshold:    {result['threshold']:.6f}\n")
    return result, store.reference_profile


def _write_session_manifest(split: str, slug: str, session_id: str) -> str:
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

    return str(manifest_path)


def main() -> None:
    args = parse_args()
    results = run_pipeline(split=args.split, scenario=args.scenario, run_all=args.all)

    try:
        get_client().flush()
    except Exception:
        pass
    try:
        langfuse_client.flush()
    except Exception:
        pass

    print(f"{len(results)} dataset runs completed.")
    print("Langfuse flush requested.")


if __name__ == "__main__":
    main()
