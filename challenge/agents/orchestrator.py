"""
OrchestratorAgent: the LLM-driven core of the multi-agent system.

The LLM decides which tools to call, in what order, and with what parameters.
It reasons step-by-step, adapts based on intermediate results, and makes the
final classification decisions for the evaluation dataset.

All LLM calls are tracked in Langfuse via @observe decorators.
"""
from __future__ import annotations
import json
from langfuse import observe, get_client
from pathlib import Path
from core.config import LevelConfig
from core.feature_store import FeatureStore
from core.llm_client import LLMClient
from tools.tool_definitions import TOOL_DEFINITIONS

SYSTEM_PROMPT = """You are the lead AI agent for a public health monitoring system called MirrorLife.

Your role: orchestrate a team of specialist tools to classify citizens as needing preventive health support (label=1) or continuing standard monitoring (label=0).

You will process one level of the challenge. Follow this workflow:

1. Load training data → inspect the summary
2. Analyze patterns to understand what signals indicate a citizen needs preventive support
3. Load evaluation data
4. Engineer features on both splits (always include 'event_features')
5. Train a classifier
6. Tune the decision threshold to maximize F1
7. Generate predictions and write the output file

Key rules:
- Reason briefly before each tool call
- Do not call all tools blindly — adapt based on what you observe
- event_features contains the strongest signal (event type escalation)
- persona_features encodes behavioral profiles: age, mobility, health risk, social isolation — use them
- After seeing CV F1 results, decide if re-training with a different model is warranted
- Minimize LLM calls and tool overhead; be efficient

Output format requirement: both output files must contain only citizen IDs (one per line) for citizens predicted as label=1.
Outputs are written to outputs/evaluation/level{N}_predictions.txt (submission) and outputs/training/level{N}_predictions.txt (reference).

When you have written both output files and are done, say "PIPELINE COMPLETE" and explain your key findings."""


class ToolDispatcher:
    """Routes tool calls from the LLM to the correct agent."""

    def __init__(self, agents: dict, store: FeatureStore, output_dir: Path):
        self.agents = agents
        self.store = store
        self.output_dir = output_dir

    def dispatch(self, tool_name: str, tool_input: dict) -> str:
        try:
            result = self._call(tool_name, tool_input)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": tool_name})

    def _call(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name == "load_data":
            return self.agents["data"].run(split=tool_input["split"])

        elif tool_name == "analyze_patterns":
            return self.agents["analysis"].run(
                analysis_type=tool_input["analysis_type"]
            )

        elif tool_name == "engineer_features":
            return self.agents["feature"].run(
                split=tool_input["split"],
                feature_groups=tool_input["feature_groups"],
            )

        elif tool_name == "train_model":
            return self.agents["training"].run(
                model_type=tool_input.get("model_type", "gradient_boosting"),
                use_class_weight=tool_input.get("use_class_weight", True),
            )

        elif tool_name == "tune_threshold":
            return self.agents["threshold"].run()

        elif tool_name == "predict_and_write":
            return self._predict_and_write(
                threshold=float(tool_input["threshold"]),
                output_path=tool_input.get("output_path", ""),
            )

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _predict_and_write(self, threshold: float, output_path: str) -> dict:
        level = self.store.metadata.get("level", 1)

        def _run_split(split: str) -> tuple[list, str]:
            features_df = self.store.get_features(split)
            X = features_df.drop(columns=["CitizenID"]).values
            citizen_ids = features_df["CitizenID"].values
            proba = self.store.model.predict_proba(X)[:, 1]
            preds = (proba >= threshold).astype(int)
            positives = [str(cid) for cid, p in zip(citizen_ids, preds) if p == 1]

            if split == "eval":
                path = output_path if output_path else str(
                    self.output_dir / "evaluation" / f"level{level}_predictions.txt"
                )
            else:
                path = str(self.output_dir / "training" / f"level{level}_predictions.txt")

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for cid in positives:
                    f.write(cid + "\n")

            return positives, path

        eval_positives, eval_path = _run_split("eval")
        train_positives, train_path = _run_split("train")

        return {
            "eval_output_path": eval_path,
            "train_output_path": train_path,
            "eval_predicted_positive_ids": eval_positives,
            "train_predicted_positive_ids": train_positives,
            "n_eval_positive": len(eval_positives),
            "n_train_positive": len(train_positives),
            "threshold_used": threshold,
        }


class OrchestratorAgent:
    MAX_TURNS = 20  # safety circuit breaker

    def __init__(
        self,
        config: LevelConfig,
        store: FeatureStore,
        llm: LLMClient,
        agents: dict,
        output_dir: Path,
    ):
        self.config = config
        self.store = store
        self.llm = llm
        self.dispatcher = ToolDispatcher(agents, store, output_dir)

    @observe(name="orchestrator")
    def run(self) -> str:
        """
        Main agentic loop. The LLM iterates through tool calls until it
        signals completion with 'PIPELINE COMPLETE'.
        """
        level = self.config.level
        self.store.metadata["level"] = level

        get_client().update_current_span(
            name=f"orchestrator_level_{level}",
            metadata={"level": level},
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Begin the classification pipeline for Level {level}. "
                    f"Training data is at: data/training/public_lev_{level}/. "
                    f"Evaluation data is at: data/evaluation/public_lev_{level}/. "
                    f"Output files: outputs/evaluation/level{level}_predictions.txt (eval) "
                    f"and outputs/training/level{level}_predictions.txt (train). "
                    "Start by loading the training data."
                ),
            }
        ]

        # Prepend system prompt as first user message (some models don't support system role)
        system_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in range(self.MAX_TURNS):
            response = self.llm.chat(
                messages=system_messages + messages,
                tools=TOOL_DEFINITIONS,
                max_tokens=1500,
            )

            content = response["content"]
            tool_calls = response["tool_calls"]

            # Add assistant response to conversation
            if tool_calls:
                # Build assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"]),
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                # Execute each tool and append results
                for tc in tool_calls:
                    print(f"[Turn {turn+1}] Tool: {tc['name']} | Input: {tc['input']}")
                    result = self.dispatcher.dispatch(tc["name"], tc["input"])
                    result_preview = result[:300] + "..." if len(result) > 300 else result
                    print(f"         Result: {result_preview}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

            else:
                # Text-only response = LLM has finished or is stuck
                messages.append({"role": "assistant", "content": content})
                print(f"[Turn {turn+1}] LLM final response: {content[:200]}")

                if "PIPELINE COMPLETE" in content:
                    break

                # If LLM stopped without completing, nudge it
                if turn < self.MAX_TURNS - 2:
                    messages.append({
                        "role": "user",
                        "content": "Continue with the next step in the pipeline.",
                    })

        return content or "Pipeline finished."
