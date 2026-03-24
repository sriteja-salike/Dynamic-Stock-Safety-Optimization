# Dynamic Safety Stock Optimization
### ML-Driven Inventory Buffer Sizing at Scale

---

## The Problem

Retailers hold safety stock — a buffer of inventory beyond expected demand — to avoid stockouts when demand spikes or suppliers run late. The traditional formula for sizing this buffer uses historical averages:

```
SS = z × √(L̄ × σ_d² + d̄² × σ_L²)
```

The formula is sound, but its inputs are not. Historical σ_d — the demand variability term — averages over years of data including every past promotion, seasonal spike, and demand shock. It cannot tell the difference between a stable March and a volatile November. It cannot see that demand has been accelerating for the past two weeks. The result is systematic over-stocking during calm periods and under-stocking ahead of volatile ones.

---

## The Approach

Replace the historical σ_d with a machine learning prediction of what demand variability will actually look like in the next 4 weeks — keeping the same proven formula but making its key input forward-looking rather than backward-looking.

| | Formula | σ_d source |
|---|---|---|
| Static (baseline) | `SS = z × √(L̄ × σ_d² + d̄² × σ_L²)` | Historical average |
| Dynamic (ML-driven) | `SS = z × √(L̄ × σ_d_ml² + d̄² × σ_L²)` | ML prediction |

Everything else in the formula stays the same. The innovation is entirely in replacing one backward-looking input with a forward-looking prediction.

**Dataset:** Walmart M5 Forecasting Competition — 5.3 years of retail sales data across 3 stores, 3,049 items (9,147 SKU-store combinations). Categories: FOODS, HOUSEHOLD, HOBBIES.

**ML Model:** A Random Forest trained on 2.2 million weekly observations to predict the standard deviation of weekly demand over the next 4 weeks. Features include rolling demand statistics, recent price changes, demand momentum, calendar signals, and retail event counts. The model achieves R²=0.44 — realistic for predicting future volatility rather than demand itself, which is an inherently harder problem.

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

81.7% of SKUs received lower safety stock recommendations — the model correctly identifying stable current conditions where the static formula was over-stocking. The remaining 18.3% received higher recommendations — the model flagging SKUs where near-term predicted variability exceeds the long-run historical average. Both directions are the correct behavior.

### By Product Category

| Category | Static Cost | Dynamic Cost | Saving |
|---|---|---|---|
| FOODS | $665.61 | $368.47 | 44.6% |
| HOUSEHOLD | $567.68 | $388.92 | 31.5% |
| HOBBIES | $265.71 | $206.84 | 22.2% |

FOODS sees the largest reduction because it has the shortest lead times — meaning demand variability dominates its safety stock calculation more than supplier reliability does. The crossover analysis below explains this precisely.

---

## Crossover Analysis

Not every SKU benefits equally from better demand forecasting. For some SKUs, lead time variability is the bigger risk — improving demand predictions barely moves the needle. For others, demand variability dominates — and that is exactly where this project's ML predictions have the highest impact.

The crossover point is the exact value of σ_d at which both terms in the formula contribute equally:

```
σ_d_crossover = d̄ × σ_L / √L̄
```

If ML-predicted σ_d > crossover → demand variability dominates → ML predictions are highly impactful  
If ML-predicted σ_d < crossover → lead time variability dominates → supplier reliability is the bigger lever

| Category | Demand-driven % | LT-driven % | Avg ML σ_d | Avg Crossover |
|---|---|---|---|---|
| FOODS | 91.4% | 8.6% | 5.06 | 3.18 |
| HOUSEHOLD | 91.0% | 9.0% | 2.99 | 1.88 |
| HOBBIES | 90.3% | 9.7% | 2.52 | 1.59 |

91% of SKUs across all categories are demand-variability-driven — confirming that ML-predicted σ_d is the highest-leverage intervention for right-sizing inventory at scale. This also directly explains the category-level savings pattern: FOODS, the most demand-driven category, saves the most.

---

## Budget Analysis

A key practical question: how much budget does a retailer actually need to implement dynamic safety stock at 95% service level?

| Budget (% of static baseline) | Weekly Budget | Feasible |
|---|---|---|
| 50% ($749) | $215 shortfall | ❌ |
| 60% ($899) | $65 shortfall | ❌ |
| 70% ($1,049) | $85 surplus | ✅ |
| 80% ($1,199) | $235 surplus | ✅ |
| 100% ($1,499) | $535 surplus | ✅ |

The dynamic approach achieves the same 95% service level at 65% of the original static budget — a $535/week saving on a portfolio of 9,147 SKUs.

---

## Business Insight

The core finding is that static safety stock systematically over-stocks because it cannot distinguish between a volatile and a stable period. Historical σ_d was 34–55% higher than the ML model's current-period prediction across all categories:

| Category | Historical σ_d | ML σ_d | Over-estimation |
|---|---|---|---|
| FOODS | 11.22 | 5.06 | 55% |
| HOUSEHOLD | 5.30 | 2.99 | 44% |
| HOBBIES | 3.81 | 2.52 | 34% |

By replacing backward-looking averages with forward-looking ML predictions, inventory planners can right-size safety stock week by week — holding less when conditions are stable, holding more when the model anticipates volatility. The same service level, at a lower cost.

---

## Interactive Tools

**Plotly Dashboard** — a scrollable visual walkthrough of the full analysis. Covers the static vs dynamic comparison, the σ_d over-estimation story, SKU-level impact distribution, crossover analysis, and budget scenarios. Standalone HTML file — opens in any browser, no setup required.

**Streamlit Budget Simulator** — a live what-if tool where the user adjusts a weekly budget via slider and sees in real time which SKUs are covered, how holding cost is allocated by category, and per-SKU detail. Answers the question: "if our inventory budget is X, how many SKUs are fully protected?" Deployed publicly and accessible from the README.

---

## What Was Explored and Why Some Components Were Dropped

Two significant methodological decisions were made during the project that are worth explaining honestly.

**Lead time ML prediction was attempted twice and abandoned.** The goal was to make both σ_d and σ_L dynamic. A carefully designed set of supplier-side features was built — supplier reliability score, order volume pressure, lead time trend, seasonal strain. Both attempts produced R²≈-0.0003 — equivalent to predicting the mean. The root cause was not the feature design but the data: lead times were simulated from a fixed distribution with no temporal structure, so no model could learn patterns that do not exist. In production with real supplier data, the feature design used here would be the correct approach. For this project, historical σ_L was used instead.

**Linear Programming was evaluated and dropped.** The original design included an LP optimizer to allocate safety stock across SKUs subject to a budget constraint. After thorough evaluation, it was found that without a binding budget constraint, LP trivially returns the service-level floor for every SKU — which is identical to direct formula computation. A budget constraint was modeled but always risks infeasibility when the constraint is tighter than the sum of service-level floors. The LP component was dropped in favor of the direct formula approach, and the budget question was addressed separately through the scenario analysis and Streamlit simulator. Recognizing when a component adds no value and removing it is itself a sound analytical decision.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · scipy · Plotly · Streamlit · matplotlib · joblib
