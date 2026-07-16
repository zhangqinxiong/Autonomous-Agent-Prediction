# 02_noskills system.md 改造设计

## 目标
将 `submissions/02_noskills/agent/prompts/system.md` 改造为多阶段自主工作流，使用 CatBoost 基线 + 特征工程 + Stacking 三阶段，agent 在预算内自主推进。

## 脚本结构

```
/work/
├── 01_baseline.py      # Phase 1: CatBoost 基线
├── 02_feature_eng.py   # Phase 2: 特征工程
├── 03_stacking.py      # Phase 3: Stacking 集成
└── submission.csv
```

## 工作流

### Step 0: EDA
- `run_command` 检查数据：shape、dtypes、缺失率、目标分布
- 记录关键发现供后续使用

### Step 1: 基线模型（01_baseline.py）

**预处理：**
1. 逐列检查缺失率：
   - 缺失 ≥ 50% → 丢弃该列
   - 缺失 < 50% → 填充：
     - 数值列 → 中位数
     - 类别列 → `"MISSING"`
2. 类别特征编码（由 agent 根据 EDA 判断）：
   - 有序特征 → `OrdinalEncoder`
   - 无序特征 → `OneHotEncoder`
   - 默认策略：低基数（≤10 类）OneHot，高基数 Ordinal

**实现方式：** 每个脚本自包含完整预处理逻辑（不互相 import），确保可独立运行。

**模型：**
- CatBoost
- `task_type="CPU"`
- `auto_class_weights="Balanced"`
- `early_stopping_rounds=50`
- `random_seed=42`, `verbose=0`
- 5-fold StratifiedKFold CV
- 每折训练，对 test 预测取平均
- 输出 CV score 和 `submission.csv`

**提交** `submit_predictions("submission.csv")`

### Step 2: 特征工程（02_feature_eng.py）
- 预算剩余时执行
- 分析 CatBoost 特征重要性
- 基于重要性/相关性构造新特征
- 复用 01_baseline.py 的预处理逻辑
- 重新训练、提交

### Step 3: Stacking（03_stacking.py）
- 预算剩余时执行
- 复用预处理 + 多模型 CV stacking
- 基础模型：CatBoost, XGBoost, LightGBM 等
- 元模型：LogisticRegression 或加权
- 提交

### Step 4: 最终选择
- `select_submission` 选择最佳 ID
- `get_status` 确认预算

## 约束
- 确保 Phase 1 至少完成一次提交
- 每一步后检查剩余 budget
- 总遵循系统限制（max_submissions, max_tool_calls, max_time）
