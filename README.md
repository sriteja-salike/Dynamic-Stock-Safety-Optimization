# Dynamic Safety Stock Optimization

ML-driven safety stock optimization using the Walmart M5 dataset. Replaces static historical demand variability estimates with forward-looking Random Forest predictions to right-size inventory buffers per SKU per period.

---

## Motivation

The standard safety stock formula uses historical averages:

```
SS = z × √(L̄ × σ_d² + d̄² × σ_L²)
```

This looks backward. It treats a quiet February the same as a volatile November. It does not adapt to current conditions.

This project replaces the historical `σ_d` (demand variability) with a ML-predicted forward-looking estimate — making safety stock adaptive to what is actually happening now, not just what happened historically.

---

## Dataset

[Kaggle M5 Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) — Walmart retail sales data

- 30,490 item-store combinations across 10 stores
- 1,941 days of data (January 2011 – June 2016)
- Supplemented with calendar event data and weekly sell prices
- **Scope:** 3 stores selected (CA_3, TX_2, WI_3) — one per state for geographic diversity

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
         ├── Simulate lead times (category-based, clipped at mean ± 3σ)
         ├── Rolling demand/LT statistics (4w, 8w, 13w)
         ├── Lag features (demand_lag1, demand_lag2)
         ├── price_change, demand_acceleration
         └── Event counts (sum of daily flags per week)
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
                       Comparison + Crossover Analysis
```

---

## ML Model — Demand Variability

**Algorithm:** RandomForestRegressor  
**Target:** Standard deviation of weekly demand over next 4 weeks  
**Features:** 26 — rolling demand statistics, lag features, price signals, event counts, calendar, encoded categoricals  
**Split:** 80/20 chronological — no shuffle

| Metric | Value |
|---|---|
| MAE | 0.9683 |
| RMSE | 2.9260 |
| R² | 0.4355 |

**Top features:** demand_std_13w (~0.52), demand_mean_4w (~0.22), demand_mean_8w (~0.06), demand_std_8w (~0.05), sell_price (~0.04)

---

## Key Results

### Dynamic vs Static Safety Stock

| Metric | Static | Dynamic | Change |
|---|---|---|---|
| Mean SS per SKU (units) | 10.71 | 6.37 | ↓ 40.5% |
| Total weekly holding cost | $1,499.00 | $964.23 | ↓ 35.7% |
| SKUs reduced | — | 7,469 | 81.7% |
| SKUs increased | — | 1,678 | 18.3% |

### By Category

| Category | Static ($) | Dynamic ($) | Saving % |
|---|---|---|---|
| FOODS | 665.61 | 368.47 | 44.6% |
| HOUSEHOLD | 567.68 | 388.92 | 31.5% |
| HOBBIES | 265.71 | 206.84 | 22.2% |

### Crossover Analysis

Crossover point per SKU: `σ_d_crossover = d̄ × σ_L / √L̄`

| Category | Demand-driven % | LT-driven % | Avg ML σ_d | Avg Crossover |
|---|---|---|---|---|
| FOODS | 91.4% | 8.6% | 5.06 | 3.18 |
| HOBBIES | 90.3% | 9.7% | 2.52 | 1.59 |
| HOUSEHOLD | 91.0% | 9.0% | 2.99 | 1.88 |

91% of SKUs are demand-variability-driven — confirming ML-predicted σ_d is the dominant lever across the portfolio.

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
├── outputs/
│   └── results_dashboard.png
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
joblib
pyarrow
```

---

## Design Decisions

- **Lead time ML abandoned:** Two attempts both produced R²≈-0.0003. Root cause: simulated lead times have no temporal structure — no learnable patterns exist. Historical per-item-store σ_L used instead.
- **LP dropped:** Without a feasible budget constraint, LP trivially returns ss_floor for every SKU — equivalent to direct formula computation. Documented as a considered and rejected component.
- **Causal feature separation:** Demand model uses only demand-side features. Lead time model used only supplier-side features. Mixing them is causally unjustified.
- **Event aggregation via sum:** Daily binary event flags summed to weekly counts — preserves full information about how many event-days occurred per week.
