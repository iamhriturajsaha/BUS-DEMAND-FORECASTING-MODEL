# 🚌 BUS DEMAND FORECASTING MODEL

## 🚀 Problem Statement

The task is to forecast **bus seat demand** at the route level exactly **15 days before the date of journey (doj)** using historical booking and search data. Demand is influenced by numerous factors such as holidays, weekends, exam seasons, and region-specific events — making this a **non-trivial time-series and tabular modeling problem**.

## 📊 Objective

Predict the `final_seatcount` (total booked seats) for each `(srcid, destid, doj)` combination using data available **15 days before the date of journey**.

---

## 📁 Dataset Description

- `train.csv`: Contains past journey-level booking information.
- `test.csv`: Contains the test journey details where predictions are needed.
- `transactions.csv`: Contains transactional data including bookings and searches on various dates.

Each record includes:
- `srcid`, `destid`, `doj`, `doi` (date of issue), `searchcount`, `seatcount`
- Region and tier information
- Derived `dbd` = (doj - doi).days

---

## 🛠️ Feature Engineering

- **Time-based features**: `dayofweek`, `month`, `is_weekend`, `is_holiday`
- **Aggregated features**: 
  - `cumsum_seatcount`, `cumsum_searchcount` (historical aggregation till `dbd = 15`)
- **Categorical mappings**: Source and destination region/tier mappings

---

## 🤖 Model Architecture

- **Model**: LightGBM Regressor
- **Validation**: GroupKFold (to prevent data leakage across routes)
- **Optimization**: Hyperparameter tuning using Optuna
- **Metric**: RMSE (Root Mean Squared Error)

---

## 📈 Performance

- Optimized for RMSE using feature-rich tabular modeling.
- Special attention given to leakage control, holiday/weekend effects, and sparse route handling.

---

## 📦 Tech Stack

- Python
- LightGBM
- Optuna
- Scikit-learn
- Pandas, NumPy

---

## 🏅 Final Rank

![Leaderboard Rank](Rank.png)

> Achieved **Rank #258** with a score of **682.60** on the private leaderboard of Analytics Vidhya's **RedBus Data Decode Hackathon 2025**.