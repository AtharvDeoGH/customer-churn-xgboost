# Customer Churn Prediction (XGBoost)

Predict customer churn using an XGBoost classifier trained on tabular behavior/engagement data. This repository includes data files, a project report, and environment requirements; add scripts/notebooks as you iterate.

## Overview
- **Goal:** Reduce churn by identifying at‑risk customers and enabling timely, targeted retention actions.
- **Team:** Analytical Aces.
- **Highlights:** High observed churn in the dataset, strong links between inactivity/visit frequency and churn, and a simple, production‑ready modeling path.

See the full slide deck in [`reports/customer-churn-xgb.pptx`](reports/customer-churn-xgb.pptx) for methodology and insights.

## Data
- Expected files:  
  - `data/train.csv` — training set  
  - `data/test.csv` — holdout / prediction set
- Typical preprocessing: numeric target encoding for `churn`, uniform scaling of selected features, retention of outliers, and creation of an `interaction_rate` feature (engagement per visit).

> **Note:** Replace file names/paths above if your structure differs.

## Methodology
- **Split:** Stratified 75/25 train/validation split.
- **Model:** XGBoost (objective: `binary:logistic`), with tuning over `eta`, `max_depth`, and `subsample`.
- **Training:** ~100 boosting rounds using DMatrix inputs.
- **Evaluation:** ROC curve and AUC; baseline validation AUC ≈ **0.75**.
- **Operationalization:** Prepare predictions for unseen data for downstream CRM/retention workflows.

## Quickstart
### 1) Setup
```bash
python -m venv .venv
# Windows: .venv\Scriptsctivate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Project skeleton
If you don’t yet have code, a simple layout works well:
```
.
├─ data/
│  ├─ train.csv
│  └─ test.csv
├─ reports/
│  └─ customer-churn-xgb.pptx
├─ src/
│  ├─ features.py
│  ├─ train.py
│  └─ predict.py
└─ requirements.txt
```

### 3) Example training snippet
> Drop this into `src/train.py` (adjust column names as needed).

```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/train.csv")
y = df["churn"].astype(int)
X = df.drop(columns=["churn"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "seed": 42,
}

bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "val")])

preds = bst.predict(dval)
print("Validation AUC:", roc_auc_score(y_val, preds))
bst.save_model("churn_xgb.json")
```

### 4) Inference example
```python
# src/predict.py
import pandas as pd
import xgboost as xgb

bst = xgb.Booster()
bst.load_model("churn_xgb.json")

X_test = pd.read_csv("data/test.csv")
dtest = xgb.DMatrix(X_test)
proba = bst.predict(dtest)
pd.DataFrame({"churn_probability": proba}).to_csv("predictions.csv", index=False)
print("Wrote predictions.csv")
```

## Results (baseline)
- Validation **AUC ≈ 0.75** on a 25% holdout (see report for details).
- Inactivity and reduced visit frequency emerge as leading churn indicators.