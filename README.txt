Reply Mirror AI Agent Challenge - Fraud Pipeline
===============================================

SETUP
-----
1. Install dependencies:
   pip install -r requirements.txt

2. Ensure the new challenge data exists under:
   challenge_data/training/
   challenge_data/evaluation/

RUN
---
Run all evaluation scenarios:
python3 pipeline.py

Run a single evaluation scenario:
python3 pipeline.py --split evaluation --scenario "Brave New World - validation"

Run all datasets in both splits:
python3 pipeline.py --all

OUTPUT
------
The pipeline writes one prediction file per dataset folder:
outputs/evaluation/<scenario_slug>_predictions.txt
outputs/training/<scenario_slug>_predictions.txt

Each file contains only suspicious transaction IDs, one per line.

ARCHITECTURE
------------
FraudOrchestrator
    |-- DataIngestionAgent     : loads transactions, users, locations, sms, mails, audio
    |-- TransactionAgent       : transaction velocity, rarity, temporal, amount features
    |-- BehaviorAgent          : location continuity, distance, city matching, behavior drift
    |-- CommunicationAgent     : phishing heuristics, communication linkage, audio activity
    |-- AnomalyAgent           : graph + unsupervised anomaly scoring
    |-- DecisionAgent          : economic weighting and adaptive thresholding

NOTES
-----
- The pipeline is deterministic and does not require an LLM to run.
- It uses every available input modality in challenge_data, including optional audio metadata.
- Thresholding is adaptive and always avoids invalid empty/all-transaction outputs.
