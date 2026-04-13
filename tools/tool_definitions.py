"""
Tool schemas (OpenAI-compatible format) that the Orchestrator LLM can call.
Each tool maps to a specific agent method.
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "load_data",
            "description": (
                "Load and merge status.csv, locations.json, and users.json for a given split. "
                "Returns a metadata summary (row counts, citizen list, event type distribution, "
                "missing-value rates). The raw data is stored internally and not returned in full."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "split": {
                        "type": "string",
                        "enum": ["train", "eval"],
                        "description": "Which dataset split to load.",
                    }
                },
                "required": ["split"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_patterns",
            "description": (
                "Run a statistical analysis on the loaded training data and return a compact "
                "text summary. Use this before deciding on features to understand data quality, "
                "label distribution, and key signals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": [
                            "event_type_distribution",
                            "biometric_trends",
                            "label_derivation",
                            "full_summary",
                        ],
                        "description": "Type of analysis to run.",
                    }
                },
                "required": ["analysis_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "engineer_features",
            "description": (
                "Compute feature groups from the loaded raw data for a given split. "
                "Available groups: event_features, biometric_features, temporal_features, user_features. "
                "Returns a summary of features computed (count, warnings). "
                "Always include 'event_features' — it contains the strongest signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "split": {
                        "type": "string",
                        "enum": ["train", "eval"],
                    },
                    "feature_groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of feature groups to compute.",
                    },
                },
                "required": ["split", "feature_groups"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": (
                "Train a classification model on the training features. "
                "Returns cross-validation F1, and the top most important features. "
                "Model is saved internally for later prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": [
                            "gradient_boosting",
                            "random_forest",
                            "logistic_regression",
                            "decision_tree",
                        ],
                        "description": "Type of classifier to train.",
                    },
                    "use_class_weight": {
                        "type": "boolean",
                        "description": "Whether to use balanced class weights for imbalanced labels.",
                    },
                },
                "required": ["model_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tune_threshold",
            "description": (
                "Sweep decision thresholds on out-of-fold predictions to find the one "
                "maximizing F1 score. Returns optimal threshold, F1 at that threshold, "
                "precision and recall breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_and_write",
            "description": (
                "Apply the trained model to the evaluation features, using the chosen threshold, "
                "and write the output .txt file. Returns the list of citizen IDs predicted as "
                "needing preventive support (label=1) and the output file path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Decision threshold between 0 and 1.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to write the output .txt file.",
                    },
                },
                "required": ["threshold", "output_path"],
            },
        },
    },
]
