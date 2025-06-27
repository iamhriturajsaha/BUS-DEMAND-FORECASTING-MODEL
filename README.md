# 🚌 BUS DEMAND FORECASTING MODEL

--- 

## 🚀 Problem Statement

Forecast the number of bus seats booked (`final_seatcount`) for intercity routes **15 days in advance** using historical booking and search data. This problem captures real-world demand trends influenced by holidays, weekends, tier/city zones, and route popularity.

---

## 🎯 Objective

Predict the `final_seatcount` for each (`srcid`, `destid`, `doj`) combination using only data available **on or before `doj - 15` (dbd < 15)**.

---

## 📁 Dataset Overview

| File               | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `train.csv`        | Historical route-level journey data with true seat counts    |
| `test.csv`         | Future journeys requiring predictions                        |
| `transactions.csv` | Transaction-level search & booking history for all journeys  |

### Key Columns
- `srcid`, `destid`: Route identifiers
- `doj`, `doi`: Date of journey / issue
- `searchcount`, `seatcount`: Metrics per transaction
- `dbd`: Days before departure = (`doj` - `doi`).days

---

## 🧠 Modeling Approach

### 🔀 Feature Engineering
- **Temporal features**: `dayofweek`, `week`, `month`, `is_weekend`
- **Aggregations (for dbd < 15 only)**:
  - Multi-level groupings: `srcid`, `destid`, `week`, `month`, `srcid+destid`
  - Stats: `mean`, `sum`, `min`, `max`, `std` on `cumsum_searchcount`, `cumsum_seatcount`
- **Lag Features**: `lag1`, `rolling_mean3`, `rolling_std3` per route
- **Categorical Encoding**: `LabelEncoder` for tier and region features

### 🧪 Validation Strategy
- **Holdout**: Nov–Dec used for validation (`doj >= 2023-11-01`)
- **GroupKFold**: Based on `doj` to avoid leakage across time
- **Metric**: RMSE

---

## 🤖 Model Architecture

We built an **ensemble of four diverse models**, each trained on different feature subsets to capture unique patterns.

| Model       | Description                                           |
|-------------|-------------------------------------------------------|
| **LightGBM**| Fast and efficient gradient boosting                  |
| **XGBoost** | Classic tree-based boosting                           |
| **CatBoost**| Handles categorical data and sparse regions well      |
| **TabNet (TABDPT)** | Deep learning for tabular data using attention  |

Each model:
- Trained using early stopping
- Tuned with consistent learning rate and depth
- Trained on individual feature subsets (`feature_sets`)
- Outputs were blended using **mean ensembling**

---

## 🔁 Ensemble Strategy

```python
# For each feature subset:
for feat_set in feature_sets:
    Train LightGBM
    Train XGBoost
    Train CatBoost
    Train TabNet
    Collect test predictions + validation predictions

# Final prediction = Average of all model outputs

---

## 🏆 Final Leaderboard Rank

Achieved **Rank #258** out of 8500+ participants by blending advanced tabular modeling with deep learning and robust validation techniques.

![Leaderboard Rank](Rank.png)



