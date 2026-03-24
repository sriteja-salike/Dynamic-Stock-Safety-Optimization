"""
Dynamic Safety Stock Optimization — Budget Simulator
Run: streamlit run app.py
Requires: sku_stats_final.parquet in the same directory
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Safety Stock Budget Simulator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  .block-container { padding-top: 2rem; padding-bottom: 2rem; }

  .kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #e2e6ed;
    border: 1px solid #e2e6ed;
    margin-bottom: 2rem;
  }
  .kpi-box {
    background: #ffffff;
    padding: 1.2rem 1.5rem;
  }
  .kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #64748b;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }
  .kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1;
  }
  .kpi-sub {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  .status-box {
    padding: 1rem 1.5rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    margin-bottom: 1.5rem;
  }
  .status-ok {
    background: rgba(22,163,74,0.08);
    border-left: 4px solid #16a34a;
    color: #15803d;
  }
  .status-no {
    background: rgba(220,38,38,0.06);
    border-left: 4px solid #dc2626;
    color: #dc2626;
  }

  .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #94a3b8;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }

  .insight {
    border-left: 3px solid #0d9488;
    padding: 0.75rem 1.2rem;
    background: rgba(13,148,136,0.04);
    font-size: 13px;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.6;
    margin-top: 1rem;
  }
  .insight strong { color: #0d9488; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet("sku_stats_final.parquet")
    return df

df = load_data()

# ── Pre-compute totals ────────────────────────────────────────────────────────
total_static  = df["static_holding_cost"].sum()
total_dynamic = df["dynamic_holding_cost"].sum()
net_saving    = total_static - total_dynamic
saving_pct    = net_saving / total_static * 100
n_skus        = len(df)

cat_order = ["FOODS", "HOUSEHOLD", "HOBBIES"]
cat_colors = {
    "FOODS":     "#0d9488",
    "HOUSEHOLD": "#2563eb",
    "HOBBIES":   "#9333ea",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📦 Dynamic Safety Stock")
    st.markdown(
        "This simulator shows the real-time impact of adjusting your inventory "
        "budget on safety stock coverage across **{:,} SKU-store combinations**.".format(n_skus)
    )
    st.divider()

    st.markdown("**Filter by category**")
    selected_cats = st.multiselect(
        label="Category",
        options=cat_order,
        default=cat_order,
        label_visibility="collapsed",
    )
    if not selected_cats:
        selected_cats = cat_order

    st.divider()
    st.markdown("**Methodology**")
    st.markdown(
        """
- Static SS uses historical σ_d
- Dynamic SS uses ML-predicted σ_d (Random Forest, R²=0.44)
- Formula: `z × √(L̄·σ_d² + d̄²·σ_L²)`
- Service level: 95% (z = 1.645)
- Dataset: Walmart M5, 3 stores
        """,
        unsafe_allow_html=False,
    )

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("## Safety Stock Budget Simulator")
st.markdown(
    "<p style='color:#64748b;font-size:14px;margin-top:-0.5rem;margin-bottom:1.5rem;'>"
    "Adjust the budget to see real-time impact on safety stock coverage and holding cost allocation."
    "</p>",
    unsafe_allow_html=True,
)

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="kpi-row">
      <div class="kpi-box">
        <div class="kpi-label">Static baseline cost</div>
        <div class="kpi-value" style="color:#64748b">${total_static:,.2f}</div>
        <div class="kpi-sub">per week · historical σ_d</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Dynamic requirement</div>
        <div class="kpi-value" style="color:#0d9488">${total_dynamic:,.2f}</div>
        <div class="kpi-sub">per week · ML-predicted σ_d</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Net saving</div>
        <div class="kpi-value" style="color:#16a34a">{saving_pct:.1f}%</div>
        <div class="kpi-sub">${net_saving:,.2f} per week</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">SKUs analysed</div>
        <div class="kpi-value" style="color:#2563eb">{n_skus:,}</div>
        <div class="kpi-sub">item-store combinations</div>
      </div>
    </div>""",
    unsafe_allow_html=True,
)

# ── Budget slider ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Budget control</div>', unsafe_allow_html=True)

budget = st.slider(
    label="Weekly inventory budget ($)",
    min_value=float(round(total_static * 0.40, 2)),
    max_value=float(round(total_static * 1.20, 2)),
    value=float(round(total_static, 2)),
    step=float(round(total_static * 0.01, 2)),
    format="$%.2f",
)

budget_pct = budget / total_static * 100
gap        = total_dynamic - budget
feasible   = gap <= 0

# ── Feasibility status ────────────────────────────────────────────────────────
if feasible:
    surplus = abs(gap)
    st.markdown(
        f"""<div class="status-box status-ok">
          ✅ &nbsp;<strong>Feasible</strong> — Budget covers the full dynamic safety stock requirement.
          &nbsp; Surplus: <strong>${surplus:,.2f}/week</strong>
          &nbsp;·&nbsp; Budget is <strong>{budget_pct:.1f}%</strong> of static baseline.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    shortfall = abs(gap)
    st.markdown(
        f"""<div class="status-box status-no">
          ❌ &nbsp;<strong>Shortfall</strong> — Budget is insufficient for 95% service level across all SKUs.
          &nbsp; Additional funding needed: <strong>${shortfall:,.2f}/week</strong>
          &nbsp;·&nbsp; Budget is <strong>{budget_pct:.1f}%</strong> of static baseline.
        </div>""",
        unsafe_allow_html=True,
    )

# ── Filter data ───────────────────────────────────────────────────────────────
df_filtered = df[df["cat_id"].isin(selected_cats)].copy()

# Compute per-SKU coverage under budget
# Proportional allocation: scale dynamic_ss down if budget is insufficient
if feasible:
    df_filtered["allocated_ss"]   = df_filtered["dynamic_ss"]
    df_filtered["allocated_cost"] = df_filtered["dynamic_holding_cost"]
    df_filtered["covered"]        = True
else:
    # Scale factor: how much of the dynamic requirement can the budget cover
    scale = budget / total_dynamic
    df_filtered["allocated_ss"]   = df_filtered["dynamic_ss"] * scale
    df_filtered["allocated_cost"] = df_filtered["dynamic_holding_cost"] * scale
    # SKU is "covered" if allocated SS >= its ss_floor (static_ss as proxy)
    df_filtered["covered"] = df_filtered["allocated_ss"] >= df_filtered["static_ss"] * 0.80

# ── Charts row ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-label">Holding cost — static vs dynamic vs allocated</div>',
                unsafe_allow_html=True)

    cat_stats = df_filtered.groupby("cat_id").agg(
        static_cost    =("static_holding_cost",  "sum"),
        dynamic_cost   =("dynamic_holding_cost", "sum"),
        allocated_cost =("allocated_cost",        "sum"),
    ).reindex([c for c in cat_order if c in selected_cats])

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        name="Static", x=cat_stats.index, y=cat_stats["static_cost"],
        marker_color="#94a3b8", marker_line_width=0,
        text=[f"${v:.0f}" for v in cat_stats["static_cost"]],
        textposition="outside", textfont=dict(size=11),
    ))
    fig_cost.add_trace(go.Bar(
        name="Dynamic (ML)", x=cat_stats.index, y=cat_stats["dynamic_cost"],
        marker_color=[cat_colors.get(c,"#0d9488") for c in cat_stats.index],
        marker_line_width=0,
        text=[f"${v:.0f}" for v in cat_stats["dynamic_cost"]],
        textposition="outside", textfont=dict(size=11),
    ))
    fig_cost.add_trace(go.Bar(
        name="Allocated (budget)", x=cat_stats.index, y=cat_stats["allocated_cost"],
        marker_color="#16a34a" if feasible else "#dc2626",
        marker_line_width=0,
        opacity=0.75,
        text=[f"${v:.0f}" for v in cat_stats["allocated_cost"]],
        textposition="outside", textfont=dict(size=11),
    ))

    fig_cost.update_layout(
        barmode="group",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        height=340, margin=dict(l=50, r=20, t=30, b=40),
        font=dict(family="IBM Plex Mono, monospace", size=11, color="#64748b"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        yaxis=dict(title="Weekly holding cost ($)", gridcolor="#edf0f5",
                   tickprefix="$"),
        xaxis=dict(gridcolor="#edf0f5"),
    )
    st.plotly_chart(fig_cost, use_container_width=True, config={"displayModeBar": False})

with col2:
    st.markdown('<div class="section-label">SKU coverage at current budget</div>',
                unsafe_allow_html=True)

    coverage = df_filtered.groupby("cat_id").agg(
        total   =("covered", "count"),
        covered =("covered", "sum"),
    ).reindex([c for c in cat_order if c in selected_cats])
    coverage["not_covered"] = coverage["total"] - coverage["covered"]
    coverage["covered_pct"] = (coverage["covered"] / coverage["total"] * 100).round(1)

    fig_cov = go.Figure()
    fig_cov.add_trace(go.Bar(
        name="Covered",
        x=coverage.index,
        y=coverage["covered"],
        marker_color="#16a34a", marker_line_width=0,
        text=[f"{p:.0f}%" for p in coverage["covered_pct"]],
        textposition="inside", textfont=dict(size=12, color="white"),
    ))
    fig_cov.add_trace(go.Bar(
        name="Not fully covered",
        x=coverage.index,
        y=coverage["not_covered"],
        marker_color="#dc2626", marker_line_width=0,
    ))

    fig_cov.update_layout(
        barmode="stack",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        height=340, margin=dict(l=50, r=20, t=30, b=40),
        font=dict(family="IBM Plex Mono, monospace", size=11, color="#64748b"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        yaxis=dict(title="Number of SKUs", gridcolor="#edf0f5"),
        xaxis=dict(gridcolor="#edf0f5"),
    )
    st.plotly_chart(fig_cov, use_container_width=True, config={"displayModeBar": False})

# ── Summary metrics row ───────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
total_covered     = int(df_filtered["covered"].sum())
total_filtered    = len(df_filtered)
alloc_cost_total  = df_filtered["allocated_cost"].sum()
alloc_ss_mean     = df_filtered["allocated_ss"].mean()

m1.metric("SKUs Covered",        f"{total_covered:,} / {total_filtered:,}")
m2.metric("Coverage Rate",       f"{total_covered/total_filtered*100:.1f}%")
m3.metric("Allocated Cost/Week", f"${alloc_cost_total:,.2f}")
m4.metric("Avg Allocated SS",    f"{alloc_ss_mean:.2f} units")

# ── SKU detail table ──────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-label">SKU-level detail</div>', unsafe_allow_html=True)

col_sort, col_top, _ = st.columns([2, 2, 4])
with col_sort:
    sort_col = st.selectbox(
        "Sort by",
        options=["holding_cost_change", "dynamic_ss", "static_ss", "allocated_ss"],
        format_func=lambda x: {
            "holding_cost_change": "Holding cost saving",
            "dynamic_ss":          "Dynamic SS",
            "static_ss":           "Static SS",
            "allocated_ss":        "Allocated SS",
        }[x],
    )
with col_top:
    top_n = st.selectbox("Show top", options=[25, 50, 100, 250], index=0)

display_df = (
    df_filtered[["item_id", "store_id", "cat_id", "static_ss", "dynamic_ss",
                 "allocated_ss", "holding_cost_change", "driver", "covered"]]
    .sort_values(sort_col, ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)

display_df.columns = [
    "Item", "Store", "Category", "Static SS",
    "Dynamic SS", "Allocated SS", "Cost Saving ($)", "Driver", "Covered"
]
display_df["Static SS"]    = display_df["Static SS"].round(2)
display_df["Dynamic SS"]   = display_df["Dynamic SS"].round(2)
display_df["Allocated SS"] = display_df["Allocated SS"].round(2)
display_df["Cost Saving ($)"] = display_df["Cost Saving ($)"].round(4)
display_df["Covered"] = display_df["Covered"].map({True: "✅", False: "❌"})

st.dataframe(
    display_df,
    use_container_width=True,
    height=400,
    hide_index=True,
)

st.markdown(
    f"""<div class="insight">
    At a weekly budget of <strong>${budget:,.2f}</strong>
    ({budget_pct:.1f}% of static baseline),
    <strong>{total_covered:,} of {total_filtered:,} SKUs</strong> meet
    their safety stock requirement — maintaining 95% service level where covered.
    The dynamic approach requires a minimum of <strong>${total_dynamic:,.2f}/week</strong>
    to cover all SKUs, versus <strong>${total_static:,.2f}/week</strong> under the static baseline.
    </div>""",
    unsafe_allow_html=True,
)
