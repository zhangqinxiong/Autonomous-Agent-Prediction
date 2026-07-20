# Autonomous Agent Prediction (Beta)

A Kaggle-in-Kaggle competition framework where LLM-powered autonomous agents act as data scientists, competing on ML datasets by writing and running code via tool-based interactions.

## Project Structure

```
├── data/                       # Evaluation harness & config
│   ├── models.yaml             # Model pricing config
│   ├── requirements.txt
│   ├── run_local_eval.py       # Local evaluation harness
│   ├── validate_submission.py  # Pre-flight validation
│   ├── sample_submission/      # Minimal sample agent
│   ├── wheels/                 # Local Python packages
│   └── kaggle-kaggle-skill/    # Evaluation utilities
├── input/                      # Competition datasets (gitignored)
│   └── train_XX/
│       ├── train.csv
│       ├── test.csv
│       ├── sample_submission.csv
│       └── solution.csv
├── output/                     # Evaluation output & traces (gitignored)
├── submissions/
│   └── 02_noskills/            # Latest submission (Public LB: 0.823)
│       ├── agent/
│       │   ├── agent.yaml      # Agent config (model, tools, prompts)
│       │   ├── prompts/
│       │   │   └── system.md   # System prompt / workflow
│       │   └── configs/
│       │       └── sampling.yaml
│       ├── switch_model.sh     # Toggle model: local ↔ kaggle
│       └── submission.zip      # Packaged for Kaggle upload
├── docs/
│   └── superpowers/            # Design docs & plans
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r data/requirements.txt

# 3. Set API keys in .env
#    GEMINI_API_KEY=...
#    DEEPSEEK_API_KEY=...
```

## Download Competition Data

```bash
python3 -c "
import kagglehub
path = kagglehub.competition_download('autonomous-agent-prediction-beta')
print('Data:', path)
"
```

## Running Locally

```bash
# Validate submission
python3 data/validate_submission.py --agent-dir submissions/02_noskills/agent

# Run evaluation on a single dataset
python3 data/run_local_eval.py \
  --submission-dir submissions/02_noskills/agent \
  --dataset train_01 \
  --metric roc_auc \
  --max-time-minutes 15 \
  --max-submissions 3

# Test pipeline directly in Kaggle Docker
docker run --rm \
  -v $(pwd):/work -w /work \
  gcr.io/kaggle-images/python:latest \
  python3 submissions/02_noskills/agent/prompts/system.md
```

### Switching Models

```bash
# Local testing (DeepSeek V3.2)
bash submissions/02_noskills/switch_model.sh local

# Kaggle submission (Gemini 3.5 Flash)
bash submissions/02_noskills/switch_model.sh kaggle
```

## Submitting to Kaggle

```bash
bash submissions/02_noskills/switch_model.sh kaggle
cd submissions/02_noskills/agent && zip -r ../submission.zip .
kaggle competitions submit autonomous-agent-prediction-beta \
  -f submissions/02_noskills/submission.zip \
  -m "message"
```

## Agent Strategy (`02_noskills`)

Current best submission scores **0.823 Public LB** on Kaggle.

### Workflow

| Phase | Description |
|-------|-------------|
| **Phase 0: EDA** | Inspect data shape, missing %, target distribution, categorical cardinality |
| **Phase 1: Baseline CatBoost** | Preprocess → CatBoost 5-fold CV → submit |
| **Phase 2: Feature Engineering** | (if budget remains) Feature importance analysis → polynomial/interaction/ratio features → retrain |
| **Phase 3: Stacking Ensemble** | (if budget remains) CatBoost + XGBoost + LightGBM with LogisticRegression meta-model |
| **Final Step** | Select best submissions for private LB via `select_submission()` |

### Preprocessing Pipeline

1. **No column dropping** — All features retained regardless of missing rate
2. **Numeric missing** → Median fill
3. **Categorical missing** → `"MISSING"` string fill
4. **Categorical encoding** → One-hot encoding (all categories, no ordinal split)

### CatBoost Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `task_type` | `"CPU"` | Kaggle environment has no GPU |
| `loss_function` | `"Logloss"` | Binary classification |
| `auto_class_weights` | `"Balanced"` | Handle imbalanced targets |
| `early_stopping_rounds` | `50` | Prevent overfitting on validation set |
| `random_seed` | `42` | Reproducibility |
| CV | 5-fold Stratified | Robust evaluation across folds |

### Local Test Results (13 datasets, CV AUC ROC)

| Dataset | Rows | Features | Cat | Train AUC | CV AUC | Gap |
|---------|------|----------|-----|-----------|--------|-----|
| train_01 | 14,957 | 12 | 5 | 0.7537 | 0.7121 | 0.0416 |
| train_02 | 14,929 | 28 | 2 | 0.9999 | 0.9698 | 0.0301 |
| train_03 | 3,501 | 18 | 7 | 0.9154 | 0.8099 | 0.1055 |
| train_04 | 8,775 | 12 | 0 | 0.9102 | 0.8297 | 0.0805 |
| train_05 | 1,060 | 9 | 5 | 0.8699 | 0.6825 | 0.1874 |
| train_06 | 10,803 | 9 | 9 | 0.8409 | 0.8111 | 0.0298 |
| train_07 | 10,417 | 17 | 6 | 0.9284 | 0.8264 | 0.1020 |
| train_08 | 8,173 | 12 | 12 | 0.8985 | 0.8545 | 0.0441 |
| train_09 | 1,109 | 18 | 4 | 0.8900 | 0.6446 | 0.2455 |
| train_10 | 11,800 | 27 | 0 | 0.9673 | 0.8438 | 0.1236 |
| train_11 | 28,879 | 20 | 0 | 0.9164 | 0.8194 | 0.0970 |
| train_12 | 49,432 | 8 | 3 | 0.8009 | 0.7868 | 0.0141 |
| train_13 | 500 | 9 | 4 | 0.9151 | 0.6728 | 0.2423 |

Small datasets (train_05, train_09, train_13 with <2K rows) show larger generalization gaps due to limited samples. Larger datasets consistently achieve strong CV scores (0.78–0.97).

### Key Design Decisions

- **Simplicity over complexity** — Default CatBoost hyperparameters outperform extensive tuning on unseen data from the same family
- **No column dropping** — Even high-missing-rate columns may carry signal; CatBoost is robust to noise
- **One-hot encoding** — All categorical columns use OHE regardless of cardinality; no ordinal/OHE split needed
- **Balanced class weights** — Ensures robustness across imbalanced datasets without manual adjustment
- **Early stopping** — Prevents overfitting while letting the model train as long as beneficial
