# 02_noskills System.md Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `submissions/02_noskills/agent/prompts/system.md` with a 3-phase workflow: CatBoost baseline → feature engineering → stacking ensemble.

**Architecture:** A single prompt file (`system.md`) that guides the competition agent through staged execution. Each phase writes and runs a self-contained Python script. The agent decides when to advance based on remaining budget.

**Tech Stack:** CatBoost (CPU, Balanced, early_stopping), pandas, numpy, scikit-learn, OrdinalEncoder/OneHotEncoder, XGBoost, LightGBM

## Global Constraints
- One file to modify: `submissions/02_noskills/agent/prompts/system.md`
- Scripts written by the agent at runtime must be self-contained (no cross-imports)
- Must ensure Phase 1 completes at least one submission
- Agent must check budget after each phase

---

### Task 1: Phase 1 — CatBoost Baseline (system.md sections)

**Files:**
- Modify: `submissions/02_noskills/agent/prompts/system.md`

**Interfaces:**
- Consumes: N/A (first task)
- Produces: system.md with EDA + Phase 1 workflow

- [ ] **Step 1: Read current system.md**

```bash
cat submissions/02_noskills/agent/prompts/system.md
```
Expected: current 127-line single-script prompt.

- [ ] **Step 2: Write new system.md header + EDA phase**

Replace the current system.md content with the new design. Write the header sections (Task, Goal, Environment, Tools, Budget) and the EDA section:

```
You are an autonomous AI data scientist in a Kaggle competition.

## Task
{problem_description}

## Goal
Maximize **{metric_name}** ({metric_direction}).

## Environment
Offline Linux container with pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, torch.
Working directory: `/work` (contains train.csv, test.csv, sample_submission.csv)

## Tools
- run_command(cmd) — Execute shell/Python commands
- write_file(path, content) — Create files
- edit_file(path, old, new) — Edit files
- submit_predictions(path) — Submit CSV for scoring
- select_submission(ids) — Select best for private LB
- get_status() — Check budget

## Budget
Use get_status() periodically. Max {max_submissions} submissions, {max_time_minutes} min.

## Workflow

Call get_status() before each phase to confirm remaining budget.

### Phase 0: Exploratory Data Analysis

Run the following command to inspect the data:
```bash
python3 -c "
import pandas as pd, numpy as np
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('Train shape:', train.shape)
print('Test shape:', test.shape)
print('--- Train dtypes ---')
print(train.dtypes.value_counts())
print('--- Missing % ---')
print((train.isnull().mean() * 100).round(1))
print('--- Target distribution ---')
target_col = train.columns[-1]
print(train[target_col].value_counts(normalize=True).round(3))
print('--- Categorical columns unique counts ---')
cat_cols = train.select_dtypes(include=['object']).columns
for c in cat_cols:
    print(f'{c}: {train[c].nunique()} unique')
"
```

Review the output. Note:
- Which columns have high missing rates (>=50% — candidates for dropping)
- Which columns have low missing rates (fill)
- Which columns are categorical and their cardinality
- Target distribution (balanced or imbalanced)
```

- [ ] **Step 3: Write Phase 1 — Baseline CatBoost section**

Add after the EDA phase:

```
### Phase 1: Baseline CatBoost

Write and run `01_baseline.py`:

```python
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
id_col, target = train.columns[0], train.columns[-1]
train_ids = train[id_col].copy()
test_ids = test[id_col].copy()
train.drop(columns=[id_col], inplace=True)
test.drop(columns=[id_col], inplace=True)

y = train[target].values
train.drop(columns=[target], inplace=True)

all_data = pd.concat([train, test]).reset_index(drop=True)

# Identify column types
numeric_cols = all_data.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = all_data.select_dtypes(include=["object"]).columns.tolist()

# Step 1: Drop columns with >=50% missing
for c in all_data.columns:
    if all_data[c].dtype in [np.float64, np.int64, np.float32, np.int32]:
        if all_data[c].isnull().mean() >= 0.5:
            all_data.drop(columns=[c], inplace=True)

# Step 2: Fill remaining missing
for c in all_data.columns:
    if all_data[c].dtype in [np.float64, np.int64, np.float32, np.int32]:
        all_data[c] = all_data[c].fillna(all_data[c].median())
    else:
        all_data[c] = all_data[c].astype(str).fillna("MISSING")

# Step 3: Encode categorical features
cat_cols = all_data.select_dtypes(include=["object"]).columns.tolist()
# Agent should inspect cardinality to decide ordering:
# For known ordinal features or high-cardinality (>10): OrdinalEncoder
# For low-cardinality (<=10) nominal features: OneHotEncoder
ordinal_cols = [c for c in cat_cols if all_data[c].nunique() > 10]
ohe_cols = [c for c in cat_cols if all_data[c].nunique() <= 10]

X_all = all_data.copy()
if ohe_cols:
    X_all = pd.get_dummies(X_all, columns=ohe_cols, drop_first=False)
if ordinal_cols:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_all[ordinal_cols] = oe.fit_transform(X_all[ordinal_cols])

n = len(train)
X = X_all.iloc[:n].values
X_test = X_all.iloc[n:].values

skf = StratifiedKFold(5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
cv_scores = []

for tr, va in skf.split(X, y):
    model = CatBoostClassifier(
        task_type="CPU",
        auto_class_weights="Balanced",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=0
    )
    model.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
    pred = model.predict_proba(X[va])[:, 1]
    cv_scores.append(roc_auc_score(y[va], pred))
    test_preds += model.predict_proba(X_test)[:, 1] / 5

print(f"CV scores: {cv_scores}")
print(f"Mean CV: {np.mean(cv_scores):.6f}")

sub = pd.DataFrame({id_col: test_ids, target: test_preds})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
```

**Important**: After writing, adjust the ordinal/OHE split. If you identified columns with clear ordinal meaning during EDA, move them to `ordinal_cols` regardless of cardinality. For anonymous competition features, use the cardinality heuristic.

Run: `python3 01_baseline.py`

Review the CV score, then submit:
```
submit_predictions("submission.csv")
```

Check the public LB score from the response.
```

- [ ] **Step 4: Write Phase 2 & 3 — Feature Engineering + Stacking sections**

Add after Phase 1:

```
### Phase 2: Feature Engineering (if budget remains)

Only proceed if you have remaining submissions and tool calls.

1. Read the feature importances from the CatBoost model in Phase 1
2. Write `02_feature_eng.py`:
   - Copy the preprocessing from `01_baseline.py`
   - Add feature engineering based on your EDA insights:
     - Polynomial features for top important numeric features
     - Feature interactions between important features
     - Binning of continuous features if appropriate
     - Aggregations or ratio features
   - Retrain CatBoost with the same CV=5 setup
   - Save and submit
3. Run, submit, check score

### Phase 3: Stacking Ensemble (if budget remains)

Only proceed if you have remaining submissions and tool calls.

Write `03_stacking.py`:
   - Copy the preprocessing from `01_baseline.py`
   - 5-fold CV training for CatBoost, XGBoost, LightGBM
   - Collect OOF predictions
   - Train LogisticRegression meta-model on OOF predictions
   - Predict on test via averaged base model predictions
   - Save submission and submit

If no budget remains for training, call:
```
select_submission(["best_submission_id"])
get_status()
```

### Final Step

Before time expires, call `select_submission()` with your best submission IDs.
```

- [ ] **Step 5: Run self-review**

Check for:
- All placeholders resolved
- Consistent terminology
- Complete workflow from start to finish
- Budget check reminders at each phase transition

- [ ] **Step 6: Save and verify**

```bash
wc -l submissions/02_noskills/agent/prompts/system.md
```
Expected: system.md exists with reasonable line count.
