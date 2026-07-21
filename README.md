# Autonomous Agent Prediction (Beta)

A Kaggle-in-Kaggle competition where LLM-powered autonomous agents act as data
scientists. Agents compete on 16 binary classification datasets by writing and
executing Python code via tool-based interactions within a sandboxed container.

**Current best: 0.823 Public LB** — Submission: `02_noskills`

---

## Project Structure

```
├── data/                           # Evaluation harness & config
│   ├── models.yaml                 # Model pricing & rate limits
│   ├── requirements.txt            # Python dependencies
│   ├── run_local_eval.py           # Local evaluation harness
│   ├── validate_submission.py      # Pre-flight validation script
│   ├── sample_submission/          # Minimal reference agent
│   ├── wheels/                     # Bundled Python packages
│   └── kaggle-kaggle-skill/       # Evaluation skill resources
├── input/                          # 16 competition datasets (gitignored)
│   └── train_XX/
│       ├── train.csv               # Training set (features + target)
│       ├── test.csv                # Test set (features only)
│       ├── sample_submission.csv   # Submission template
│       └── solution.csv            # Ground truth (Public/Private split)
├── submissions/
│   └── 02_noskills/                # Current best submission
│       ├── agent/
│       │   ├── agent.yaml          # Agent config (model, tools, prompt)
│       │   ├── prompts/
│       │   │   └── system.md       # System prompt / workflow definition
│       │   └── configs/
│       │       └── sampling.yaml   # Generation parameters
│       ├── switch_model.sh         # Toggle: local ↔ Kaggle model
│       └── submission.zip         # Packaged for Kaggle upload
├── docs/
│   └── superpowers/                # Design documents & plans
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r data/requirements.txt

# 3. Configure API keys in .env
cat > .env << EOF
GEMINI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
EOF
```

### Download Competition Data

```bash
python3 -c "
import kagglehub
path = kagglehub.competition_download('autonomous-agent-prediction-beta')
print('Data downloaded to:', path)
"
```

---

## Local Evaluation

```bash
# Validate submission structure
python3 data/validate_submission.py --agent-dir submissions/02_noskills/agent

# Run agent against a single dataset
python3 data/run_local_eval.py \
  --submission-dir submissions/02_noskills/agent \
  --dataset train_01 \
  --metric roc_auc \
  --max-time-minutes 15 \
  --max-submissions 3

# Test in Kaggle Docker environment
docker run --rm \
  -v $(pwd):/work -w /work \
  gcr.io/kaggle-images/python:latest \
  python3 submissions/02_noskills/agent/prompts/system.md
```

### Switch Model

```bash
# Local testing (DeepSeek V3.2)
bash submissions/02_noskills/switch_model.sh local

# Kaggle submission (Gemini 3.5 Flash)
bash submissions/02_noskills/switch_model.sh kaggle

# Check current model
bash submissions/02_noskills/switch_model.sh
```

### Baseline Pipeline (Local Test Script)

Run the ensemble locally to reproduce CV scores:

```bash
python3 baseline.py
```

This runs CatBoost + XGBoost + LightGBM with 5-fold CV + target encoding across
all 15 datasets (excluding train_14), outputting per-model CV, ensemble CV, and
Public/Private/Full LB scores.

---

## Submitting to Kaggle

```bash
# 1. Switch to Kaggle model
bash submissions/02_noskills/switch_model.sh kaggle

# 2. Package
cd submissions/02_noskills/agent
zip -r ../submission.zip agent.yaml prompts/ configs/ \
  -x "*/.ipynb_checkpoints/*"

# 3. Submit
cd ../..
kaggle competitions submit autonomous-agent-prediction-beta \
  -f submissions/02_noskills/submission.zip \
  -m "description of changes"

# 4. Check status
kaggle competitions submissions autonomous-agent-prediction-beta | head -10
```

---

## Agent Strategy: `02_noskills`

### Workflow

| Phase | Description |
|-------|-------------|
| **Phase 0: EDA** | Inspect shape, dtypes, missing %, target distribution, categorical cardinality |
| **Phase 1: Baseline Ensemble** | Median fill + target encoding (within CV folds) → CatBoost + XGBoost + LightGBM 5-fold CV → Dirichlet ensemble blending → submit |
| **Phase 2: Feature Engineering** | (budget permitting) Feature importance → polynomial/interaction/ratio features → retrain ensemble |
| **Phase 3: Standalone Model** | (budget permitting) Hyperparameter tuning or alternative approach |
| **Final Step** | `select_submission()` with best IDs, `get_status()` to confirm |

### Preprocessing

1. **Numeric features** — Median fill (computed on train+test jointly)
2. **Categorical features** — `"MISSING"` string fill, then **target encoding** within each CV fold (smoothed with global mean, smooth=20)
3. **No one-hot encoding** — Target encoding produces numeric features directly
4. **No column dropping** — All features retained

### Ensemble Configuration

All three models use GPU (`device="cuda"` or `task_type="GPU"`), 3000 max
iterations, learning_rate=0.03, max_depth=6, 5-fold StratifiedKFold, and
early_stopping_rounds=50.

| Model | Class Weight | Key Params |
|-------|-------------|------------|
| **CatBoost** | `auto_class_weights="Balanced"` | `loss_function="Logloss"`, symmetric trees |
| **XGBoost** | `scale_pos_weight` (auto-computed) | `device="cuda"` |
| **LightGBM** | `class_weight="balanced"` | `device="gpu"` |

Ensemble weights are found via 5000 Dirichlet random samples maximizing OOF
ROC-AUC.

### Local Test Results (15 datasets)

| Dataset | Rows | Num | Cat | CB CV | XGB CV | LGB CV | Ensemble | Public | Private |
|---------|------|-----|-----|-------|--------|--------|----------|--------|---------|
| train_01 | 14,957 | 7 | 5 | 0.7125 | 0.7072 | 0.7089 | 0.7126 | 0.7175 | 0.7103 |
| train_02 | 14,929 | 26 | 2 | 0.9703 | 0.9638 | 0.9635 | 0.9704 | 0.9736 | 0.9716 |
| train_03 | 3,501 | 11 | 7 | 0.8104 | 0.7939 | 0.7972 | 0.8104 | 0.8094 | 0.8233 |
| train_04 | 8,775 | 12 | 0 | 0.8307 | 0.8268 | 0.8271 | 0.8318 | 0.8351 | 0.8358 |
| train_05 | 1,060 | 4 | 5 | 0.6767 | 0.6611 | 0.6656 | 0.6791 | 0.6759 | 0.6889 |
| train_06 | 10,803 | 0 | 9 | 0.8102 | 0.8085 | 0.8102 | 0.8109 | 0.8005 | 0.8144 |
| train_07 | 10,417 | 11 | 6 | 0.8265 | 0.8215 | 0.8204 | 0.8278 | 0.8330 | 0.8335 |
| train_08 | 8,173 | 0 | 12 | 0.8547 | 0.8497 | 0.8513 | 0.8547 | 0.8553 | 0.8502 |
| train_09 | 1,109 | 14 | 4 | 0.6421 | 0.6050 | 0.6160 | 0.6429 | 0.6353 | 0.6536 |
| train_10 | 11,800 | 27 | 0 | 0.8443 | 0.8320 | 0.8329 | 0.8450 | 0.8405 | 0.8523 |
| train_11 | 28,879 | 20 | 0 | 0.8219 | 0.8122 | 0.8115 | 0.8225 | 0.8357 | 0.8239 |
| train_12 | 49,432 | 5 | 3 | 0.7870 | 0.7858 | 0.7866 | 0.7875 | 0.7955 | 0.7724 |
| train_13 | 500 | 5 | 4 | 0.6631 | 0.6352 | 0.6494 | 0.6646 | 0.6405 | 0.6535 |
| train_15 | 500 | 6 | 24 | 0.8551 | 0.8276 | 0.8313 | 0.8561 | 0.8557 | 0.8599 |
| train_16 | 1,809 | 21 | 0 | 0.9056 | 0.8742 | 0.8707 | 0.9056 | 0.9171 | 0.9109 |
| **Mean** | | | | **0.8007** | **0.7869** | **0.7895** | **0.8015** | **0.8014** | **0.8036** |

### Key Design Decisions

- **Target encoding over one-hot** — Reduces dimensionality for high-cardinality
  categoricals; applied within CV folds to prevent leakage
- **Three-model ensemble** — CatBoost dominates (avg weight ~0.85), but
  XGBoost/LightGBM provide diversity on specific datasets
- **GPU acceleration** — ~10x faster tree building on RTX 2080 Ti
- **No feature selection** — Tree-based models handle irrelevant features
  natively
- **No scaling** — Tree models are scale-invariant
- **Balanced class weights** — Ensures robustness across imbalanced datasets
  without per-dataset tuning
- **Early stopping** — Prevents overfitting while adapting to each dataset's
  optimal iteration count

---

## Notes

- **train_14 is excluded** from all tests (reserved as a held-out challenge set)
- The agent runs inside a Kaggle container **without GPU** — local GPU results
  serve as approximate upper bounds
- `catboost_info/` is a runtime artifact that may be root-owned and can be safely
  ignored
