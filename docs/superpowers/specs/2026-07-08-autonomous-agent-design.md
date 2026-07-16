# Autonomous ML Agent Design for Kaggle-in-Kaggle Competition

## Overview
Build an autonomous ML agent for the Kaggle-in-Kaggle competition that reads tabular datasets, trains XGBoost models with automated hyperparameter tuning, and submits predictions — all without human intervention.

## Architecture

### Model
- **LLM**: DeepSeek V3.2 (via LiteLLM)
- **ML Engine**: XGBoost with automated hyperparameter tuning

### Submission Structure
```
submissions/01_baseline/agent/
├── agent.yaml                    # ADK Agent config
├── prompts/
│   └── system.md                 # System instruction for LLM orchestration
├── configs/
│   └── sampling.yaml             # LLM generation parameters
└── skills/
    └── ml_workflow/
        ├── SKILL.md              # Skill manifest
        └── scripts/
            ├── 01_eda.py         # Exploratory data analysis
            ├── 02_preprocess.py  # Preprocessing + feature engineering
            ├── 03_train.py       # XGBoost training + hyperparameter tuning
            └── 04_predict.py     # Generate predictions
```

### Workflow
1. **EDA**: Explore data types, missing values, distributions
2. **Preprocess**: Handle missing values, encode categoricals, scale numerics
3. **Train**: XGBoost with automated param tuning + cross-validation
4. **Predict**: Generate submission CSV
5. **Submit**: Use submit_predictions tool
6. **Select**: Choose best submission via select_submission

### Budget Strategy
- Periodically call `get_status()` to monitor budget
- Prioritize: EDA → Model Training → Multiple Submissions → Selection
- Default to best public score if select_submission is not called

## ML Pipeline Details

### Preprocessing
- Numeric features: median imputation, standard scaling
- Categorical features: label encoding, mode imputation for missing
- Ordinal features: ordinal encoding preserving order

### Training (03_train.py)
- XGBoost with hyperparameter search:
  - n_estimators: [100, 200, 500]
  - max_depth: [3, 5, 7, 9]
  - learning_rate: [0.01, 0.05, 0.1]
  - subsample: [0.6, 0.8, 1.0]
  - colsample_bytree: [0.6, 0.8, 1.0]
- 5-fold StratifiedKFold cross-validation
- Early stopping with 50 rounds
- Metric: ROC AUC
