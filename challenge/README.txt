Reply Mirror AI Agent Challenge - Solution
==========================================

SUBMISSION CHECKLIST
---------------------
1. Run:  python3 pipeline.py [level]     (1, 2, or 3)
2. Note the SESSION ID printed at the end
3. Submit:
   a. Langfuse Session ID (shown in terminal output)
   b. Output file: outputs/level{N}_predictions.txt
   c. Source code .zip (for evaluation datasets only)

SETUP
-----
1. Add your OPENROUTER_API_KEY to .env:
   Get a free key at https://openrouter.ai

2. Install dependencies:
   pip install -r requirements.txt

3. Data must be in:
   data/training/public_lev_{1,2,3}/
   data/evaluation/public_lev_{1,2,3}/

RUN
---
python3 pipeline.py 1     # Level 1
python3 pipeline.py 2     # Level 2
python3 pipeline.py 3     # Level 3

# Check session traces:
python3 main.py <session_id>

OUTPUT FILES (already generated)
---------------------------------
outputs/level1_predictions.txt  -> VGYMNRLD
outputs/level2_predictions.txt  -> RLREEIPQ, KZCGTADH, HICNSNAE
outputs/level3_predictions.txt  -> SXGNVXTY, HGGNBOGI, JBMQBKLY, UGKAYQBB, ZXFUOOCO

ARCHITECTURE
------------
OrchestratorAgent (LLM via OpenRouter)
    |-- DataIngestionAgent      : loads status.csv, locations.json, users.json
    |-- PatternAnalysisAgent    : identifies at-risk signals in training data
    |-- FeatureEngineeringAgent : event/biometric/temporal/user features (43 total)
    |-- ModelTrainingAgent      : GradientBoosting with stratified CV, F1=1.0
    |-- ThresholdTuningAgent    : sweeps thresholds to maximize F1

Key insight: Citizens with escalated event types (emergency visit,
specialist consultation, follow-up assessment) are label=1.
The LLM discovers this from training data and orchestrates accordingly.

Langfuse tracking: https://challenges.reply.com/langfuse
Credentials in .env (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)