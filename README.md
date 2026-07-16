# Autonomous Agent Prediction (Beta)

A Kaggle-in-Kaggle competition framework where LLM-powered autonomous agents act as data scientists, competing on ML datasets by writing and running code via tool-based interactions.

## Project Structure

```
├── data/
│   ├── data/                    # Competition datasets (train_01 ~ train_16)
│   │   └── train_XX/
│   │       ├── train.csv
│   │       ├── test.csv
│   │       ├── sample_submission.csv
│   │       └── solution.csv
│   ├── models.yaml              # Model pricing config
│   ├── requirements.txt
│   ├── run_local_eval.py        # Local evaluation harness
│   ├── validate_submission.py   # Pre-flight validation
│   ├── sample_submission/       # Minimal sample agent
│   ├── wheels/                  # Local Python packages
│   └── kaggle-kaggle-skill/     # Evaluation utilities
├── submissions/
│   └── 02_noskills/             # Latest submission
│       ├── agent/
│       │   ├── agent.yaml       # Agent config (model, tools, prompts)
│       │   ├── prompts/
│       │   │   └── system.md    # System prompt / workflow
│       │   └── configs/
│       │       └── sampling.yaml
│       ├── output/              # Evaluation traces
│       ├── submission.zip       # Packaged for Kaggle upload
│       └── switch_model.sh      # Toggle model: local ↔ kaggle
├── docs/
│   └── superpowers/             # Design docs & plans
├── .env                         # LLM API keys
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

## Running Locally

```bash
# Validate submission
cd data
python validate_submission.py --agent-dir ../submissions/02_noskills/agent

# Run evaluation on a single dataset
cd data
python run_local_eval.py \
  --submission-dir ../submissions/02_noskills/agent \
  --dataset train_01 \
  --metric roc_auc \
  --max-time-minutes 15 \
  --max-submissions 3
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

## Agent Workflow (`02_noskills`)

The agent follows a 3-phase workflow within a single session:

| Phase | Script | Description |
|-------|--------|-------------|
| **Phase 0** | EDA | Inspect data shape, missing %, target distribution, categorical cardinality |
| **Phase 1** | `01_baseline.py` | Drop ≥50% missing cols → median/"MISSING" fill → OHE + Ordinal encoding → CatBoost (CPU, Balanced, early_stopping) → 5-fold CV → submit |
| **Phase 2** | `02_feature_eng.py` | Analyze feature importance → create polynomial/interaction/ratio features → retrain → submit |
| **Phase 3** | `03_stacking.py` | CatBoost + XGBoost + LightGBM CV stacking → LogisticRegression meta-model → submit |

### Preprocessing
- Columns with ≥50% missing values → dropped
- Remaining numeric columns → median fill
- Categorical columns → "MISSING" fill
- Low-cardinality (≤10) features → OneHotEncoder
- High-cardinality (>10) or ordinal features → OrdinalEncoder
