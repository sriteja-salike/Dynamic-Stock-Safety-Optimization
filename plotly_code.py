"""
Dynamic Safety Stock Optimization — Interactive Dashboard
Place in same directory as sku_stats_final.parquet and run.
Outputs: dashboard.html (standalone, no server needed)
"""

import pandas as pd
import numpy as np
import json

df = pd.read_parquet("sku_stats_final.parquet")

# ── Pre-compute ───────────────────────────────────────────────────────────────
total_static  = df["static_holding_cost"].sum()
total_dynamic = df["dynamic_holding_cost"].sum()
net_saving    = total_static - total_dynamic
saving_pct    = net_saving / total_static * 100
n_reduced     = int((df["ss_change"] < 0).sum())
n_increased   = int((df["ss_change"] > 0).sum())

cat_summary = df.groupby("cat_id").agg(
    static_total         =("static_holding_cost", "sum"),
    dynamic_total        =("dynamic_holding_cost","sum"),
    avg_sigma_d_hist     =("sigma_d",             "mean"),
    avg_sigma_d_ml       =("sigma_d_ml",          "mean"),
    avg_crossover        =("sigma_d_crossover",   "mean"),
).reset_index()
cat_summary["saving_pct"] = (
    (cat_summary["static_total"] - cat_summary["dynamic_total"])
    / cat_summary["static_total"] * 100
)
# Ensure consistent order
cat_order = ["FOODS","HOUSEHOLD","HOBBIES"]
cat_summary = cat_summary.set_index("cat_id").loc[cat_order].reset_index()

cat_driver = (
    df.groupby(["cat_id","driver"]).size()
    .unstack(fill_value=0).reset_index()
)
cat_driver = cat_driver.set_index("cat_id").loc[cat_order].reset_index()
cat_driver["total"]      = cat_driver[["demand_driven","lt_driven"]].sum(axis=1)
cat_driver["demand_pct"] = (cat_driver["demand_driven"] / cat_driver["total"] * 100).round(1)
cat_driver["lt_pct"]     = (cat_driver["lt_driven"]     / cat_driver["total"] * 100).round(1)

budget_factors = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]
budget_vals    = [round(total_static * f, 2) for f in budget_factors]
budget_labels  = [f"{int(f*100)}%" for f in budget_factors]

# Histogram bins for SS % change
pct_changes = df["ss_change_pct"].values
bin_edges = np.arange(
    np.floor(pct_changes.min() / 10) * 10,
    np.ceil(pct_changes.max() / 10) * 10 + 10,
    10
)
hist_counts, hist_edges = np.histogram(pct_changes, bins=bin_edges)
hist_labels = [str(int(e)) for e in hist_edges[:-1]]
hist_colors = ["#16a34a" if e < 0 else "#dc2626" for e in hist_edges[:-1]]

# Crossover scatter — sample 2000 points for performance
sample = df.sample(min(2000, len(df)), random_state=42)
scatter_data = {}
for cat in cat_order:
    sub = sample[sample["cat_id"] == cat]
    scatter_data[cat] = [
        {"x": round(float(r["sigma_d_crossover"]),3),
         "y": round(float(r["sigma_d_ml"]),3),
         "item": r["item_id"],
         "store": r["store_id"],
         "static_ss": round(float(r["static_ss"]),2),
         "dynamic_ss": round(float(r["dynamic_ss"]),2),
         "driver": r["driver"]}
        for _, r in sub.iterrows()
    ]

scatter_max = float(max(sample["sigma_d_crossover"].max(), sample["sigma_d_ml"].max())) * 1.08

# JSON for JS
j = lambda x: json.dumps(x)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dynamic Safety Stock Optimization</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#f8f9fb;--panel:#ffffff;--border:#e2e6ed;
  --text:#1a202c;--muted:#64748b;--grid:#edf0f5;
}}
html{{scroll-behavior:smooth}}
body{{background:var(--bg);color:var(--text);font-family:'IBM Plex Sans',sans-serif;font-size:15px;line-height:1.65}}

nav{{
  position:fixed;top:0;left:0;right:0;height:50px;
  background:rgba(248,249,251,0.96);backdrop-filter:blur(10px);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 2.5rem;gap:2rem;z-index:100;
}}
nav .brand{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;color:#0d9488;letter-spacing:.1em;text-transform:uppercase;margin-right:auto}}
nav a{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);text-decoration:none;letter-spacing:.04em;transition:color .15s}}
nav a:hover{{color:var(--text)}}

.hero{{min-height:80vh;display:flex;flex-direction:column;justify-content:center;padding:6rem 4rem 4rem;border-bottom:1px solid var(--border)}}
.hero-tag{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#0d9488;letter-spacing:.18em;text-transform:uppercase;margin-bottom:1.25rem}}
.hero h1{{font-family:'IBM Plex Sans',sans-serif;font-size:clamp(1.5rem,2.8vw,2.3rem);font-weight:400;line-height:1.2;max-width:660px;margin-bottom:1.2rem;letter-spacing:-.02em}}
.hero h1 em{{font-style:normal;color:#0d9488;font-weight:600}}
.hero-desc{{font-size:14px;color:var(--muted);max-width:540px;margin-bottom:3rem;font-weight:300}}

.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);max-width:800px;border:1px solid var(--border)}}
.kpi{{padding:1.2rem 1.5rem;background:var(--panel);border-right:1px solid var(--border)}}
.kpi:last-child{{border-right:none}}
.kpi-label{{font-family:'IBM Plex Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:.14em;text-transform:uppercase;margin-bottom:.35rem}}
.kpi-value{{font-family:'IBM Plex Mono',monospace;font-size:1.65rem;font-weight:600;line-height:1}}
.kpi-sub{{font-size:11px;color:var(--muted);margin-top:.3rem;font-family:'IBM Plex Mono',monospace}}

.section{{padding:4rem 4rem;border-bottom:1px solid var(--border);max-width:1260px;margin:0 auto}}
.section h2{{font-size:clamp(1.15rem,1.8vw,1.5rem);font-weight:400;letter-spacing:-.02em;margin-bottom:.6rem}}
.section-desc{{color:var(--muted);max-width:660px;margin-bottom:2rem;font-size:14px;font-weight:300}}

.chart-wrap{{background:var(--panel);border:1px solid var(--border);padding:1.5rem}}
.chart-title{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:1rem}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
.two-col-form{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem}}

.formula-block{{background:var(--panel);border:1px solid var(--border);padding:1.2rem 1.6rem;font-family:'IBM Plex Mono',monospace}}
.formula-block .flabel{{font-size:9px;color:var(--muted);letter-spacing:.14em;text-transform:uppercase;margin-bottom:.4rem}}
.formula-block .ftext{{font-size:14px;color:var(--text)}}
.formula-block .ftext .hl{{color:#0d9488;font-weight:600}}
.formula-block .ftext .dim{{color:var(--muted)}}
.formula-block .fsub{{font-size:11px;color:var(--muted);margin-top:.3rem}}

.insight{{border-left:3px solid #0d9488;padding:.85rem 1.3rem;background:rgba(13,148,136,.04);margin-top:1.2rem;font-size:13px;color:var(--muted);font-family:'IBM Plex Mono',monospace;line-height:1.6}}
.insight strong{{color:#0d9488}}

.leg{{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:12px;font-size:12px;color:var(--muted);font-family:'IBM Plex Mono',monospace}}
.leg-item{{display:flex;align-items:center;gap:5px}}
.leg-swatch{{width:10px;height:10px;border-radius:2px}}

.data-table{{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:12px;margin-top:1.5rem;background:var(--panel);border:1px solid var(--border)}}
.data-table th{{text-align:left;padding:.6rem 1rem;font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);background:var(--bg)}}
.data-table td{{padding:.65rem 1rem;border-bottom:1px solid var(--border)}}
.data-table tr:last-child td{{border-bottom:none}}
.data-table tr:hover td{{background:rgba(13,148,136,.03)}}
.tag{{display:inline-block;padding:2px 8px;font-size:10px}}
.tag-ok{{background:rgba(22,163,74,.1);color:#15803d}}
.tag-no{{background:rgba(220,38,38,.08);color:#dc2626}}

@media(max-width:900px){{
  .hero{{padding:5rem 1.5rem 3rem}}
  .section{{padding:3rem 1.5rem}}
  .two-col,.two-col-form{{grid-template-columns:1fr}}
  .kpi-grid{{grid-template-columns:1fr 1fr}}
  .kpi{{border-bottom:1px solid var(--border)}}
  nav a{{display:none}}
}}
</style>
</head>
<body>

<nav>
  <span class="brand">Dynamic Safety Stock · Optimization</span>
  <a href="#problem">Problem</a>
  <a href="#variability">Variability</a>
  <a href="#distribution">Distribution</a>
  <a href="#crossover">Crossover</a>
  <a href="#budget">Budget</a>
</nav>

<!-- HERO -->
<section class="hero">
  <div class="hero-tag">Supply Chain · Inventory Optimization · Machine Learning</div>
  <h1>Dynamic Safety Stock via<br><em>ML-Predicted</em> Demand Variability</h1>
  <p class="hero-desc">
    Replacing static historical demand variability with forward-looking Random Forest predictions
    to right-size safety stock across 9,147 SKU-store combinations — Walmart M5 dataset.
  </p>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Cost Reduction</div>
      <div class="kpi-value" style="color:#0d9488">{saving_pct:.1f}%</div>
      <div class="kpi-sub">${total_static:.0f} → ${total_dynamic:.0f}/wk</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">SKUs Reduced</div>
      <div class="kpi-value" style="color:#2563eb">{n_reduced:,}</div>
      <div class="kpi-sub">{n_reduced/len(df)*100:.1f}% of portfolio</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">SKUs Increased</div>
      <div class="kpi-value" style="color:#9333ea">{n_increased:,}</div>
      <div class="kpi-sub">higher near-term risk</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Min Budget · 95% SL</div>
      <div class="kpi-value" style="color:#d97706">${total_dynamic:.0f}</div>
      <div class="kpi-sub">65% of static baseline</div>
    </div>
  </div>
</section>

<!-- SECTION 1 -->
<section class="section" id="problem">
  <h2>The Problem with Static Safety Stock</h2>
  <p class="section-desc">
    The standard formula uses historical σ_d averaged over years of data including all volatile periods.
    It cannot distinguish a stable March from a volatile November — leading to systematic over-stocking.
  </p>
  <div class="two-col-form">
    <div class="formula-block">
      <div class="flabel">Static Formula — Baseline</div>
      <div class="ftext">SS = z × √( L̄ × <span class="dim">σ_d²</span> + d̄² × σ_L² )</div>
      <div class="fsub">σ_d = historical average — never adapts</div>
    </div>
    <div class="formula-block">
      <div class="flabel">Dynamic Formula — ML-Driven</div>
      <div class="ftext">SS = z × √( L̄ × <span class="hl">σ_d_ml²</span> + d̄² × σ_L² )</div>
      <div class="fsub">σ_d_ml = RF prediction of next-4-week variability</div>
    </div>
  </div>

  <div class="chart-wrap">
    <div class="chart-title">Total weekly holding cost — static vs dynamic</div>
    <div class="leg">
      <span class="leg-item"><span class="leg-swatch" style="background:#94a3b8"></span>Static (historical)</span>
      <span class="leg-item"><span class="leg-swatch" style="background:#0d9488"></span>FOODS dynamic</span>
      <span class="leg-item"><span class="leg-swatch" style="background:#2563eb"></span>HOUSEHOLD dynamic</span>
      <span class="leg-item"><span class="leg-swatch" style="background:#9333ea"></span>HOBBIES dynamic</span>
    </div>
    <div style="position:relative;width:100%;height:320px">
      <canvas id="fig1"></canvas>
    </div>
  </div>

  <div class="insight">
    <strong>FOODS ↓{cat_summary.loc[cat_summary.cat_id=='FOODS','saving_pct'].values[0]:.1f}%</strong> &nbsp;·&nbsp;
    <strong>HOUSEHOLD ↓{cat_summary.loc[cat_summary.cat_id=='HOUSEHOLD','saving_pct'].values[0]:.1f}%</strong> &nbsp;·&nbsp;
    <strong>HOBBIES ↓{cat_summary.loc[cat_summary.cat_id=='HOBBIES','saving_pct'].values[0]:.1f}%</strong>
    — variance in savings reflects each category's lead time profile and demand structure.
  </div>
</section>

<!-- SECTION 2 -->
<section class="section" id="variability">
  <h2>Why Static Over-Stocks: σ_d Comparison</h2>
  <p class="section-desc">
    Historical σ_d averages over all past volatile periods. The ML model predicts σ_d for the current
    period only — correctly identifying it as a stable, non-promotional window.
    The crossover point marks where lead time variability begins to dominate over demand variability.
  </p>

  <div class="chart-wrap">
    <div class="chart-title">Avg demand variability (σ_d) — historical vs ML-predicted vs crossover</div>
    <div class="leg">
      <span class="leg-item"><span class="leg-swatch" style="background:#94a3b8"></span>Historical σ_d</span>
      <span class="leg-item"><span class="leg-swatch" style="background:#ea580c"></span>ML-predicted σ_d</span>
      <span class="leg-item"><span class="leg-swatch" style="background:#d97706"></span>Crossover threshold</span>
    </div>
    <div style="position:relative;width:100%;height:300px">
      <canvas id="fig2"></canvas>
    </div>
  </div>

  <div class="insight">
    Historical σ_d exceeded ML predictions by
    <strong>{((cat_summary.loc[cat_summary.cat_id=='FOODS','avg_sigma_d_hist'].values[0] - cat_summary.loc[cat_summary.cat_id=='FOODS','avg_sigma_d_ml'].values[0]) / cat_summary.loc[cat_summary.cat_id=='FOODS','avg_sigma_d_hist'].values[0] * 100):.0f}% for FOODS</strong>,
    <strong>{((cat_summary.loc[cat_summary.cat_id=='HOUSEHOLD','avg_sigma_d_hist'].values[0] - cat_summary.loc[cat_summary.cat_id=='HOUSEHOLD','avg_sigma_d_ml'].values[0]) / cat_summary.loc[cat_summary.cat_id=='HOUSEHOLD','avg_sigma_d_hist'].values[0] * 100):.0f}% for HOUSEHOLD</strong>, and
    <strong>{((cat_summary.loc[cat_summary.cat_id=='HOBBIES','avg_sigma_d_hist'].values[0] - cat_summary.loc[cat_summary.cat_id=='HOBBIES','avg_sigma_d_ml'].values[0]) / cat_summary.loc[cat_summary.cat_id=='HOBBIES','avg_sigma_d_hist'].values[0] * 100):.0f}% for HOBBIES</strong>.
    The static formula has been over-stocking during stable periods for years.
  </div>
</section>

<!-- SECTION 3 -->
<section class="section" id="distribution">
  <h2>SKU-Level Impact Distribution</h2>
  <p class="section-desc">
    {n_reduced/len(df)*100:.1f}% of SKUs received lower safety stock recommendations.
    {n_increased/len(df)*100:.1f}% received higher — the model correctly flagging SKUs where
    near-term predicted variability exceeds the historical average.
  </p>

  <div class="chart-wrap">
    <div class="chart-title">Distribution of safety stock % change per SKU (dynamic vs static)</div>
    <div class="leg">
      <span class="leg-item"><span class="leg-swatch" style="background:#16a34a"></span>Reduced (ML sees lower variability ahead)</span>
      <span class="leg-item"><span class="leg-swatch" style="background:#dc2626"></span>Increased (ML flags higher variability ahead)</span>
    </div>
    <div style="position:relative;width:100%;height:300px">
      <canvas id="fig3"></canvas>
    </div>
  </div>

  <div class="insight">
    The spread confirms the model is <strong>genuinely differentiating between SKUs</strong> —
    not uniformly cutting buffers. SKUs in red were correctly identified as higher near-term risk.
  </div>
</section>

<!-- SECTION 4 -->
<section class="section" id="crossover">
  <h2>Crossover Analysis — Demand vs Lead Time Driver</h2>
  <p class="section-desc">
    The crossover point (σ_d_crossover = d̄ × σ_L / √L̄) marks where lead time variability begins
    to dominate. Points above the diagonal are demand-driven — ML predictions have the highest leverage there.
    Points below are lead-time-driven — supplier reliability matters more.
  </p>

  <div class="two-col">
    <div class="chart-wrap">
      <div class="chart-title">σ_d_ml vs crossover point per SKU — hover for details</div>
      <div class="leg">
        <span class="leg-item"><span class="leg-swatch" style="background:#808080"></span>FOODS</span>
        <span class="leg-item"><span class="leg-swatch" style="background:#ffd700"></span>HOUSEHOLD</span>
        <span class="leg-item"><span class="leg-swatch" style="background:#4b0082"></span>HOBBIES</span>
        <span class="leg-item"><span class="leg-swatch" style="background:#d97706;height:3px;width:18px;border-radius:0"></span>Crossover line</span>
      </div>
      <div id="fig4" style="width:100%;height:340px"></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">% of SKUs — demand-driven vs LT-driven by category</div>
      <div class="leg">
        <span class="leg-item"><span class="leg-swatch" style="background:#0891b2"></span>Demand-driven</span>
        <span class="leg-item"><span class="leg-swatch" style="background:#94a3b8"></span>Lead time-driven</span>
      </div>
      <div style="position:relative;width:100%;height:300px">
        <canvas id="fig5"></canvas>
      </div>
    </div>
  </div>

  <div class="insight">
    <strong>{int((df['driver']=='demand_driven').mean()*100)}% of SKUs are demand-variability-driven</strong> across all categories —
    confirming ML-predicted σ_d is the highest-leverage intervention for right-sizing inventory at scale.
  </div>
</section>

<!-- SECTION 5 -->
<section class="section" id="budget">
  <h2>Budget Scenario Analysis</h2>
  <p class="section-desc">
    How much budget is needed to achieve 95% service level using dynamic safety stock?
    Green bars meet or exceed the dynamic requirement. Red bars fall short.
    Any allocation at or above 65% of the static baseline fully covers all 9,147 SKUs.
  </p>

  <div class="chart-wrap">
    <div class="chart-title">Weekly budget vs dynamic safety stock requirement (${ total_dynamic:.0f}/week for 95% service level)</div>
    <div class="leg">
      <span class="leg-item"><span class="leg-swatch" style="background:#16a34a"></span>Feasible (meets 95% SL)</span>
      <span class="leg-item"><span class="leg-swatch" style="background:#dc2626"></span>Shortfall</span>
      <span class="leg-item"><span style="width:18px;height:2px;background:#1a202c;display:inline-block;border-bottom:2px dashed #1a202c"></span>&nbsp;Dynamic requirement</span>
    </div>
    <div style="position:relative;width:100%;height:320px">
      <canvas id="fig6"></canvas>
    </div>
  </div>

  <table class="data-table">
    <thead>
      <tr>
        <th>Budget Factor</th>
        <th>Weekly Budget ($)</th>
        <th>vs Dynamic Requirement (${total_dynamic:.0f})</th>
        <th>95% SL Achievable</th>
      </tr>
    </thead>
    <tbody>
      {''.join(f"""<tr>
        <td>{int(f*100)}% of static baseline</td>
        <td>${total_static*f:,.2f}</td>
        <td style="color:{'#dc2626' if total_dynamic-(total_static*f) > 0 else '#15803d'}">
          ${abs(total_dynamic-(total_static*f)):,.2f} {"shortfall" if total_dynamic-(total_static*f) > 0 else "surplus"}</td>
        <td><span class="tag {'tag-no' if total_dynamic-(total_static*f) > 0 else 'tag-ok'}">
          {"No" if total_dynamic-(total_static*f) > 0 else "Yes"}</span></td>
      </tr>""" for f in budget_factors)}
    </tbody>
  </table>

  <div class="insight">
    Dynamic SS achieves <strong>95% service level at ${total_dynamic:.0f}/week</strong> —
    saving <strong>${net_saving:.2f} ({saving_pct:.1f}%)</strong> versus the static baseline of ${total_static:.0f}/week.
  </div>
</section>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
const gridColor = '#edf0f5';
const textColor = '#64748b';

const baseOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, font: {{ size: 11 }} }} }},
    y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, font: {{ size: 11 }} }} }}
  }}
}};

// ── FIG 1 — Holding Cost ──────────────────────────────────────────────────────
new Chart(document.getElementById('fig1'), {{
  type: 'bar',
  data: {{
    labels: {j(cat_order)},
    datasets: [
      {{
        label: 'Static',
        data: {j([round(float(cat_summary.loc[cat_summary.cat_id==c,'static_total'].values[0]),2) for c in cat_order])},
        backgroundColor: '#94a3b8',
        borderRadius: 4, borderSkipped: false,
      }},
      {{
        label: 'Dynamic',
        data: {j([round(float(cat_summary.loc[cat_summary.cat_id==c,'dynamic_total'].values[0]),2) for c in cat_order])},
        backgroundColor: ['#0d9488','#2563eb','#9333ea'],
        borderRadius: 4, borderSkipped: false,
      }}
    ]
  }},
  options: {{
    ...baseOpts,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.raw.toFixed(2)}}` }} }}
    }},
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ color: textColor, font: {{ size: 12 }} }} }},
      y: {{
        grid: {{ color: gridColor }},
        ticks: {{ color: textColor, callback: v => '$' + v }},
        title: {{ display: true, text: 'Total weekly holding cost ($)', color: textColor, font: {{ size: 11 }} }}
      }}
    }}
  }}
}});

// ── FIG 2 — σ_d Comparison ───────────────────────────────────────────────────
new Chart(document.getElementById('fig2'), {{
  type: 'bar',
  data: {{
    labels: {j(cat_order)},
    datasets: [
      {{
        label: 'Historical σ_d',
        data: {j([round(float(cat_summary.loc[cat_summary.cat_id==c,'avg_sigma_d_hist'].values[0]),3) for c in cat_order])},
        backgroundColor: '#94a3b8', borderRadius: 4, borderSkipped: false,
      }},
      {{
        label: 'ML-predicted σ_d',
        data: {j([round(float(cat_summary.loc[cat_summary.cat_id==c,'avg_sigma_d_ml'].values[0]),3) for c in cat_order])},
        backgroundColor: '#ea580c', borderRadius: 4, borderSkipped: false,
      }},
      {{
        label: 'Crossover threshold',
        data: {j([round(float(cat_summary.loc[cat_summary.cat_id==c,'avg_crossover'].values[0]),3) for c in cat_order])},
        backgroundColor: '#d97706', borderRadius: 4, borderSkipped: false,
      }}
    ]
  }},
  options: {{
    ...baseOpts,
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ color: textColor, font: {{ size: 12 }} }} }},
      y: {{
        grid: {{ color: gridColor }},
        ticks: {{ color: textColor }},
        title: {{ display: true, text: 'Avg demand variability (σ_d)', color: textColor, font: {{ size: 11 }} }}
      }}
    }}
  }}
}});

// ── FIG 3 — Histogram ─────────────────────────────────────────────────────────
new Chart(document.getElementById('fig3'), {{
  type: 'bar',
  data: {{
    labels: {j(hist_labels)},
    datasets: [{{
      data: {j([int(x) for x in hist_counts])},
      backgroundColor: {j(hist_colors)},
      borderRadius: 2, borderSkipped: false,
      barPercentage: 1.0, categoryPercentage: 1.0,
    }}]
  }},
  options: {{
    ...baseOpts,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{
        title: ctx => `SS change: ${{ctx[0].label}}`,
        label: ctx => ` ${{ctx.raw.toLocaleString()}} SKUs`
      }} }}
    }},
    scales: {{
      x: {{
        grid: {{ display: false }},
        ticks: {{ color: textColor, font: {{ size: 10 }}, maxRotation: 0, autoSkip: false }},
        title: {{ display: true, text: 'Safety stock change (%)', color: textColor, font: {{ size: 11 }} }}
      }},
      y: {{
        grid: {{ color: gridColor }},
        ticks: {{ color: textColor }},
        title: {{ display: true, text: 'Number of SKUs', color: textColor, font: {{ size: 11 }} }}
      }}
    }}
  }}
}});

// ── FIG 4 — Crossover Scatter (Plotly for hover) ──────────────────────────────
const scatterData = {j(scatter_data)};
const scatterMax = {round(scatter_max, 2)};
const catColors = {{'FOODS':'rgba(128,128,128,0.6)','HOUSEHOLD':'rgba(255,215,0,0.7)','HOBBIES':'rgba(75,0,130,0.7)'}};

const traces = Object.entries(scatterData).map(([cat, pts]) => ({{
  x: pts.map(p => p.x),
  y: pts.map(p => p.y),
  mode: 'markers',
  type: 'scatter',
  name: cat,
  marker: {{ color: catColors[cat], size: 5 }},
  customdata: pts.map(p => [p.item, p.store, p.static_ss, p.dynamic_ss, p.driver]),
  hovertemplate:
    '<b>%{{customdata[0]}} @ %{{customdata[1]}}</b><br>' +
    'σ_d_ml: %{{y:.2f}}<br>Crossover: %{{x:.2f}}<br>' +
    'Static SS: %{{customdata[2]}}<br>Dynamic SS: %{{customdata[3]}}<br>' +
    'Driver: %{{customdata[4]}}<extra></extra>'
}}));

traces.push({{
  x: [0, scatterMax], y: [0, scatterMax],
  mode: 'lines', type: 'scatter', name: 'Crossover line',
  line: {{ color: '#d97706', dash: 'dash', width: 1.5 }},
  hoverinfo: 'skip', showlegend: false
}});

Plotly.newPlot('fig4', traces, {{
  paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
  margin: {{ l: 50, r: 20, t: 20, b: 50 }},
  xaxis: {{
    title: {{ text: 'σ_d crossover point', font: {{ size: 11, color: '#64748b' }} }},
    gridcolor: '#edf0f5', zerolinecolor: '#edf0f5', range: [0, scatterMax]
  }},
  yaxis: {{
    title: {{ text: 'ML-predicted σ_d', font: {{ size: 11, color: '#64748b' }} }},
    gridcolor: '#edf0f5', zerolinecolor: '#edf0f5', range: [0, scatterMax]
  }},
  font: {{ family: 'IBM Plex Mono, monospace', color: '#64748b', size: 11 }},
  showlegend: false,
  annotations: [
    {{ x: scatterMax*0.5, y: scatterMax*0.82, text: '▲ Demand-driven', showarrow: false, font: {{ color: '#0d9488', size: 11 }} }},
    {{ x: scatterMax*0.65, y: scatterMax*0.28, text: '▼ LT-driven', showarrow: false, font: {{ color: '#d97706', size: 11 }} }}
  ]
}}, {{ displayModeBar: false, responsive: true }});

// ── FIG 5 — Driver Classification ────────────────────────────────────────────
new Chart(document.getElementById('fig5'), {{
  type: 'bar',
  data: {{
    labels: {j(cat_order)},
    datasets: [
      {{
        label: 'Demand-driven',
        data: {j([round(float(cat_driver.loc[cat_driver.cat_id==c,'demand_pct'].values[0]),1) for c in cat_order])},
        backgroundColor: '#0891b2', borderRadius: 4, borderSkipped: false,
      }},
      {{
        label: 'LT-driven',
        data: {j([round(float(cat_driver.loc[cat_driver.cat_id==c,'lt_pct'].values[0]),1) for c in cat_order])},
        backgroundColor: '#94a3b8', borderRadius: 4, borderSkipped: false,
      }}
    ]
  }},
  options: {{
    ...baseOpts,
    scales: {{
      x: {{ stacked: true, grid: {{ display: false }}, ticks: {{ color: textColor, font: {{ size: 12 }} }} }},
      y: {{
        stacked: true, min: 0, max: 100,
        grid: {{ color: gridColor }},
        ticks: {{ color: textColor, callback: v => v + '%' }},
        title: {{ display: true, text: '% of SKUs', color: textColor, font: {{ size: 11 }} }}
      }}
    }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.raw}}%` }} }}
    }}
  }}
}});

// ── FIG 6 — Budget Scenario ───────────────────────────────────────────────────
const dynamicReq = {round(total_dynamic, 2)};
const budgetVals = {j(budget_vals)};
const budgetLabels = {j(budget_labels)};

new Chart(document.getElementById('fig6'), {{
  type: 'bar',
  data: {{
    labels: budgetLabels,
    datasets: [{{
      label: 'Weekly budget',
      data: budgetVals,
      backgroundColor: budgetVals.map(v => v >= dynamicReq ? '#16a34a' : '#dc2626'),
      borderRadius: 4, borderSkipped: false,
    }}]
  }},
  plugins: [{{
    id: 'reqLine',
    afterDraw(chart) {{
      const {{ctx, scales: {{x, y}}}} = chart;
      const yPx = y.getPixelForValue(dynamicReq);
      ctx.save();
      ctx.strokeStyle = '#1a202c';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(x.left, yPx);
      ctx.lineTo(x.right, yPx);
      ctx.stroke();
      ctx.fillStyle = '#1a202c';
      ctx.font = '11px IBM Plex Mono, monospace';
      ctx.fillText('95% SL requirement: $' + dynamicReq.toFixed(0), x.right - 210, yPx - 6);
      ctx.restore();
    }}
  }}],
  options: {{
    ...baseOpts,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => {{
            const gap = ctx.raw - dynamicReq;
            return gap >= 0
              ? ` ${{ctx.raw.toFixed(2)}} — surplus ${{gap.toFixed(2)}}`
              : ` ${{ctx.raw.toFixed(2)}} — shortfall ${{Math.abs(gap).toFixed(2)}}`;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{
        grid: {{ display: false }},
        ticks: {{ color: textColor }},
        title: {{ display: true, text: 'Budget as % of static baseline', color: textColor, font: {{ size: 11 }} }}
      }},
      y: {{
        grid: {{ color: gridColor }},
        ticks: {{ color: textColor, callback: v => '$' + v }},
        title: {{ display: true, text: 'Weekly budget ($)', color: textColor, font: {{ size: 11 }} }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

with open("plotly_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("plotly_dashboard.html saved — open in any browser.")
