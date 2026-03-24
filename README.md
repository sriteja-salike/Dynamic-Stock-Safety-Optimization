[![View Plotly Dashboard](https://img.shields.io/badge/View%20Plotly%20Dashboard-0d9488?style=for-the-badge)](https://sriteja-salike.github.io/Dynamic-Stock-Safety-Optimization/plotly_dashboard.html)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dynamic-stock-safety.streamlit.app/)

# Dynamic Safety Stock Optimization

ML-driven safety stock optimization using the Walmart M5 dataset. Replaces static historical demand variability estimates with forward-looking Random Forest predictions to right-size inventory buffers per SKU per period.

---

## Motivation

The standard safety stock formula uses historical averages:

```
SS = z × √(L̄ × σ_d² + d̄² × σ_L²)
```

This looks backward. It treats a quiet February the same as a volatile November. It does not adapt to current conditions.

This project replaces the historical `σ_d` (demand variability) with a ML-predicted forward-looking estimate — making safety stock adaptive to what is actually happening now, not just what happened historically. `σ_L` uses historical per-item-store std; ML-based lead time prediction was attempted and abandoned — see Design Decisions.

---

## Dataset

[Kaggle M5 Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) — Walmart retail sales data

- 30,490 item-store combinations across 10 stores
- 1,941 days of data (January 2011 – June 2016)
- Supplemented with calendar event data and weekly sell prices
- **Scope:** 3 stores selected (CA_3, TX_2, WI_3) — one per state for geographic diversity, chosen by highest total demand and lowest zero-demand percentage per state

---

## Pipeline

```
Raw M5 Data
    │
    ├── Melt wide → long format
    ├── Merge calendar (events) + sell_prices
    └── Filter to 3 stores
         │
         ▼
    Weekly Aggregation (df_weekly)
         │
         ├── Simulate lead times (category-based, clipped min=1d, max=mean+3σ per category)
         ├── Rolling demand/LT statistics (4w, 8w, 13w) — all shifted 1 week to prevent leakage
         ├── Lag features (demand_lag1, demand_lag2)
         ├── price_change, demand_acceleration
         └── Event counts (sum of daily binary flags per week)
              │
              ▼
         Feature Split
              │
              ├── df_model_demand (26 features → target: σ_d next 4 weeks)
              │        │
              │        └── RandomForestRegressor → model_demand_std.pkl
              │
              └── df_model_lt (17 features → target: σ_L next 4 weeks)
                       │
                       └── FAILED (R²≈-0.0003) → historical σ_L used instead
                            │
                            ▼
                       Static SS Baseline
                       (historical σ_d, historical σ_L)
                            │
                            ▼
                       Dynamic SS
                       (ML-predicted σ_d, historical σ_L)
                            │
                            ▼
                       Comparison + Crossover Analysis + Budget Simulator
```

---

## Lead Time Simulation

Lead time data is not available in the M5 dataset — it is proprietary to Walmart. Lead times were simulated using category-based normal distributions grounded in domain knowledge, clipped to prevent physically unrealistic outlier values.

| Category | Mean (days) | Std (days) | Min (days) | Max (days) |
|---|---|---|---|---|
| FOODS | 3 | 1.0 | 1 | 6 |
| HOUSEHOLD | 7 | 2.0 | 1 | 13 |
| HOBBIES | 10 | 3.0 | 1 | 19 |

Fixed seed: `np.random.default_rng(seed=42)`. All formula calculations use `lead_time_weeks = lead_time_days / 7`.

---

## Feature Engineering

**Demand model features (26 total):**
- Rolling demand statistics at 4w, 8w, 13w windows (mean, std, CV) — all shifted 1 week
- Lag features: `demand_lag1`, `demand_lag2`
- Price signal: `price_change` (pct_change of sell_price per item-store)
- Momentum: `demand_acceleration` (demand_lag1 − demand_lag2)
- Event counts: sporting, national, religious, cultural event days per week (sum of daily flags)
- Calendar: week_of_year, month, quarter
- Encoded categoricals: cat_id, dept_id, store_id, state_id

Lead time rolling features were explicitly excluded from the demand model — no causal mechanism connects supplier delivery timing to customer demand variability.

---

## ML Model — Demand Variability

**Algorithm:** RandomForestRegressor  
**Target:** Standard deviation of weekly demand over next 4 weeks  
**Features:** 26 — see Feature Engineering above  
**Split:** 80/20 chronological — no shuffle

| Metric | Value |
|---|---|
| MAE | 0.9683 |
| RMSE | 2.9260 |
| R² | 0.4355 |

**Top features by importance:** demand_std_13w (~0.52), demand_mean_4w (~0.22), demand_mean_8w (~0.06), demand_std_8w (~0.05), sell_price (~0.04)

R² of 0.44 is realistic for this problem type — predicting future variability (not demand itself) is inherently harder. The model correctly identifies stable vs volatile periods, which is what drives the safety stock reduction.

---

## Key Results

### Dynamic vs Static Safety Stock

| Metric | Static | Dynamic | Change |
|---|---|---|---|
| Mean SS per SKU (units) | 10.71 | 6.37 | ↓ 40.5% |
| Total weekly holding cost | $1,499.00 | $964.23 | ↓ 35.7% |
| SKUs reduced | — | 7,469 | 81.7% |
| SKUs increased | — | 1,678 | 18.3% |

81.7% of SKUs received lower safety stock recommendations. 18.3% received higher — the model correctly flagging SKUs where near-term predicted variability exceeds the historical average.

### By Category

| Category | Static ($) | Dynamic ($) | Saving % |
|---|---|---|---|
| FOODS | 665.61 | 368.47 | 44.6% |
| HOUSEHOLD | 567.68 | 388.92 | 31.5% |
| HOBBIES | 265.71 | 206.84 | 22.2% |

### Crossover Analysis

For each SKU, the crossover point identifies whether demand variability or lead time variability dominates the safety stock requirement:

```
σ_d_crossover = d̄ × σ_L / √L̄
```

If ML-predicted σ_d > crossover → demand-driven → ML predictions have highest leverage  
If ML-predicted σ_d < crossover → LT-driven → supplier reliability is the bigger lever

| Category | Demand-driven % | LT-driven % | Avg ML σ_d | Avg Crossover |
|---|---|---|---|---|
| FOODS | 91.4% | 8.6% | 5.06 | 3.18 |
| HOUSEHOLD | 91.0% | 9.0% | 2.99 | 1.88 |
| HOBBIES | 90.3% | 9.7% | 2.52 | 1.59 |

91% of SKUs are demand-variability-driven — confirming ML-predicted σ_d is the dominant lever across the portfolio.

### Budget Analysis

The dynamic approach requires $964.23/week to achieve 95% service level across all 9,147 SKUs — 65% of the static baseline of $1,499/week.

| Budget (% of static) | Weekly Budget | Feasible |
|---|---|---|
| 50% | $749.50 | ❌ |
| 60% | $899.40 | ❌ |
| 70% | $1,049.30 | ✅ |
| 80% | $1,199.20 | ✅ |
| 100% | $1,499.00 | ✅ |

---

## Interactive Tools

**Plotly Dashboard** — storytelling walkthrough of the full analysis: static vs dynamic comparison, σ_d breakdown, SKU distribution, crossover scatter, budget scenarios. Standalone HTML, no server required.

**Streamlit Budget Simulator** — live what-if tool. Adjust the weekly budget via slider and see real-time impact on SKU coverage, holding cost allocation by category, and per-SKU detail table. Deployed at [dynamic-stock-safety.streamlit.app](https://dynamic-stock-safety.streamlit.app/).

---

## Repository Structure

```
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_weekly_aggregation.ipynb
│   ├── 03_lead_time_simulation.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_demand_model.ipynb
│   ├── 06_lead_time_model.ipynb
│   ├── 07_static_baseline.ipynb
│   ├── 08_dynamic_ss.ipynb
│   └── 09_results_analysis.ipynb
├── models/
│   └── model_demand_std.pkl
├── data/
│   └── sku_stats_final.parquet
├── app.py
├── build_dashboard.py
├── plotly_dashboard.html
├── requirements.txt
└── README.md
```

---

## Dependencies

```
pandas
numpy
scikit-learn
scipy
matplotlib
plotly
streamlit
joblib
pyarrow
```

---

## Design Decisions

**Lead time ML abandoned:** Two separate attempts with causally designed supplier-side features (supplier reliability score, order pressure, category lead time trend, seasonal strain) both produced R²≈-0.0003. Root cause: simulated lead times have no temporal structure by construction — no learnable patterns exist. Historical per-item-store σ_L used instead.

**LP dropped:** Without a binding budget constraint, LP trivially returns ss_floor for every SKU — equivalent to direct formula computation. A budget constraint was evaluated but always risks infeasibility unless set above the ML floor cost. Documented as a considered and rejected component.

**Causal feature separation:** Demand model uses only demand-side features. Lead time features were explicitly removed after confirming near-zero importance and a marginal performance improvement upon removal (R² 0.4332 → 0.4355). Mixing causally unrelated features adds noise without adding signal.

**Event aggregation via sum:** Daily binary event flags summed to weekly counts. `first` misses mid-week events; `max` misses multiple same-type events in one week. `sum` preserves the full count of event-days per week.

**Lead time upper bound:** Initial simulation had no upper bound, producing outlier values (e.g. 24 days for HOBBIES, 4.67σ above mean). Upper clip of mean + 3σ per category added after identification. All downstream features rebuilt.
