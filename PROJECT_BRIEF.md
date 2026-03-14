# Dynamic Safety Stock Optimization
### ML-Driven Inventory Buffer Sizing at Scale

---

## The Problem

Retailers like Walmart hold safety stock — a buffer of inventory beyond expected demand — to avoid stockouts when demand spikes or suppliers run late. The traditional formula uses historical averages to size this buffer. It does not adapt to current conditions.

A stable March gets the same safety stock as a volatile November. A SKU with accelerating demand gets the same buffer as one that has been flat for weeks. This leads to systematic over-stocking during calm periods and under-stocking during volatile ones.

---

## The Approach

Replace the historical demand variability estimate in the safety stock formula with a **machine learning prediction** of what variability will look like in the next 4 weeks — making safety stock adaptive to current conditions.

| | Formula | σ_d source |
|---|---|---|
| Static (baseline) | `SS = z × √(L̄ × σ_d² + d̄² × σ_L²)` | Historical average |
| Dynamic (ML-driven) | `SS = z × √(L̄ × σ_d_ml² + d̄² × σ_L²)` | ML prediction |

**Dataset:** Walmart M5 Forecasting Competition — 5.3 years of retail sales data, 3 stores, 3,049 items (9,147 SKU-store combinations)

**ML Model:** Random Forest trained to predict the standard deviation of weekly demand over the next 4 weeks using 26 features: rolling demand statistics, price signals, event counts, lag features, and calendar variables.

| Metric | Value |
|---|---|
| R² | 0.44 |
| MAE | 0.97 |
| RMSE | 2.93 |

---

## Results

### Holding Cost Reduction

| | Static Baseline | Dynamic ML | Change |
|---|---|---|---|
| Mean safety stock per SKU | 10.71 units | 6.37 units | ↓ 40.5% |
| Total weekly holding cost | $1,499 | $964 | ↓ 35.7% |

81.7% of SKUs saw reduced safety stock recommendations. 18.3% saw increases — correctly identifying SKUs where near-term volatility is expected to exceed the historical average.

### By Product Category

| Category | Static Cost | Dynamic Cost | Saving |
|---|---|---|---|
| FOODS | $665.61 | $368.47 | 44.6% |
| HOUSEHOLD | $567.68 | $388.92 | 31.5% |
| HOBBIES | $265.71 | $206.84 | 22.2% |

FOODS benefits most because it is the most demand-variability-driven category — which the crossover analysis confirms.

---

## Crossover Analysis

For each SKU, the crossover point identifies which factor — demand variability or lead time variability — dominates the safety stock requirement:

`σ_d_crossover = d̄ × σ_L / √L̄`

If ML-predicted σ_d > crossover → demand-driven → better forecasting is the right lever  
If ML-predicted σ_d < crossover → lead-time-driven → supplier reliability is the right lever

| Category | Demand-driven % | LT-driven % | Avg ML σ_d | Avg Crossover |
|---|---|---|---|---|
| FOODS | 91.4% | 8.6% | 5.06 | 3.18 |
| HOUSEHOLD | 91.0% | 9.0% | 2.99 | 1.88 |
| HOBBIES | 90.3% | 9.7% | 2.52 | 1.59 |

**91% of SKUs across all categories are demand-variability-driven** — confirming that ML-predicted σ_d is the dominant lever for right-sizing safety stock across the portfolio.

---

## Funding Gap Analysis

How much budget is needed to achieve 95% service level using dynamic safety stock?

| Budget (% of static baseline) | Required vs Available | Feasible |
|---|---|---|
| 50% ($749) | $215 shortfall | ❌ |
| 60% ($899) | $65 shortfall | ❌ |
| 70% ($1,049) | $85 surplus | ✅ |
| 80% ($1,199) | $235 surplus | ✅ |
| 100% ($1,499) | $535 surplus | ✅ |

**The dynamic approach achieves the same 95% service level at just 65% of the original static budget.**

---

## Business Insight

The static formula systematically over-stocks because it cannot distinguish between a volatile and a stable period. Historical σ_d across all categories was 34–55% higher than the ML-predicted current-period estimate:

| Category | Historical σ_d | ML σ_d | Over-estimation |
|---|---|---|---|
| FOODS | 11.22 | 5.06 | 55% |
| HOUSEHOLD | 5.30 | 2.99 | 44% |
| HOBBIES | 3.81 | 2.52 | 34% |

By replacing backward-looking averages with forward-looking ML predictions, inventory planners can right-size safety stock week by week — holding less when conditions are stable, holding more when volatility is anticipated.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · scipy · matplotlib · joblib

---

## What Was Explored and Why Some Components Were Dropped

**Lead time ML prediction:** Two attempts using causally designed supplier-side features both failed (R²≈-0.0003). Root cause: simulated lead times have no learnable temporal structure. Historical per-SKU σ_L used instead.

**Linear Programming:** Evaluated across multiple formulations. Without a feasible budget constraint, LP trivially returns the service-level floor for every SKU — equivalent to direct formula computation. Dropped as a considered and principled decision.
