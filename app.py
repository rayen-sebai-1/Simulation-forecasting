"""
app.py
------
Streamlit dashboard for the enriched 6G network slicing time-series dataset.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="6G Network Slicing Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (minimal, professional) ───────────────────────────────────────

st.markdown(
    """
    <style>
        /* Sidebar */
        section[data-testid="stSidebar"] { background: #0f172a; }
        section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

        /* Main background */
        .main { background: #f8fafc; }

        /* Metric cards */
        [data-testid="metric-container"] {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,.07);
        }
        [data-testid="stMetricLabel"]  { font-size: 0.78rem !important; font-weight: 600; color: #ffffff !important; }
        [data-testid="stMetricValue"]  { font-size: 1.7rem  !important; color: #ffffff    !important; }

        /* Section headers */
        h2 { color: #1e293b; font-weight: 700; }

        /* Congestion badge */
        .badge-ok   { display:inline-block; padding:2px 10px; border-radius:999px;
                       background:#d1fae5; color:#065f46; font-weight:600; font-size:.8rem; }
        .badge-high { display:inline-block; padding:2px 10px; border-radius:999px;
                       background:#fee2e2; color:#991b1b; font-weight:600; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────

ENRICHED_CSV = Path("data/network_slicing_dataset_enriched_timeseries.csv")


@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if not ENRICHED_CSV.exists():
    st.error(
        f"**Enriched dataset not found** at `{ENRICHED_CSV}`.\n\n"
        "Please run first:\n```\npython enrich_6g.py\n```"
    )
    st.stop()

df_full = load_data(ENRICHED_CSV)

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Simple_SVG_logo.svg/240px-Simple_SVG_logo.svg.png",
        width=40,
    ) if False else None  # placeholder if you have a logo

    st.markdown("## 📡 6G Network Slicing")
    st.markdown("---")

    # Slice selection
    all_slices = sorted(df_full["slice_id"].unique())
    selected_slice = st.selectbox("Select Slice ID", all_slices, index=0)

    st.markdown("---")

    # Time range
    df_slice = df_full[df_full["slice_id"] == selected_slice].copy()
    t_min = df_slice["timestamp"].min().to_pydatetime()
    t_max = df_slice["timestamp"].max().to_pydatetime()

    st.markdown("#### Time Range")
    start_dt = st.slider(
        "Start",
        min_value=t_min,
        max_value=t_max,
        value=t_min,
        format="MM/DD HH:mm",
        label_visibility="collapsed",
    )
    end_dt = st.slider(
        "End",
        min_value=t_min,
        max_value=t_max,
        value=t_max,
        format="MM/DD HH:mm",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#94a3b8'>Dataset: 6G synthetic network slicing<br>"
        "Enriched with synthetic time-series resource signals</small>",
        unsafe_allow_html=True,
    )

# ── Filter data ───────────────────────────────────────────────────────────────

mask = (
    (df_full["slice_id"] == selected_slice)
    & (df_full["timestamp"] >= start_dt)
    & (df_full["timestamp"] <= end_dt)
)
df = df_full[mask].copy()

if df.empty:
    st.warning("No data in the selected time range. Adjust the sliders.")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────

slice_type = df["slice_type"].iloc[0] if "slice_type" in df.columns else selected_slice

st.markdown(
    f"<h2>6G Network Slicing — <span style='color:#3b82f6'>{selected_slice}</span> "
    f"<span style='font-size:1rem; color:#64748b; font-weight:400'>({slice_type})</span></h2>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<small style='color:#94a3b8'>Period: **{start_dt.strftime('%Y-%m-%d %H:%M')}** → "
    f"**{end_dt.strftime('%Y-%m-%d %H:%M')}** &nbsp;|&nbsp; "
    f"{len(df):,} minutes</small>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Summary metrics ───────────────────────────────────────────────────────────

avg_cpu = df["cpu_util_pct"].mean()
avg_bw  = df["bw_util_pct"].mean()
pct_congested = df["congestion_flag"].mean() * 100
max_queue = int(df["queue_len"].max())
avg_mem  = df["mem_util_pct"].mean()
avg_users = df["active_users"].mean()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("⚙️ Avg CPU %",       f"{avg_cpu:.1f}%")
c2.metric("📶 Avg BW %",        f"{avg_bw:.1f}%")
c3.metric("🧠 Avg Mem %",        f"{avg_mem:.1f}%")
c4.metric("👥 Avg Users",        f"{avg_users:.0f}")
c5.metric("🔴 Congestion %",    f"{pct_congested:.1f}%")
c6.metric("📦 Max Queue Len",   str(max_queue))

st.markdown(" ")

# ── Plot helpers ──────────────────────────────────────────────────────────────

# height is intentionally omitted from PLOT_LAYOUT so individual charts can
# override it without triggering a "multiple values" TypeError.
PLOT_LAYOUT = dict(
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    hovermode="x unified",
    font=dict(family="Inter, Arial, sans-serif", size=12, color="black"),
)


def hex_to_rgba(hex_color: str, alpha: float = 0.10) -> str:
    """Convert a 6-digit hex colour string to an rgba() string accepted by Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def line_chart(
    x, y, name: str, color: str, y_label: str, y_range=None
) -> go.Figure:
    if color.startswith("rgb"):
        fill_color = color.replace(")", ", 0.08)").replace("rgb", "rgba")
    else:
        fill_color = hex_to_rgba(color, alpha=0.10)

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=1.8),
            fill="tozeroy",
            fillcolor=fill_color,
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=280,
        yaxis_title=y_label,
        yaxis_range=y_range,
        yaxis_showgrid=True,
        yaxis_gridcolor="#f1f5f9",
        yaxis_linecolor="#e2e8f0",
        showlegend=False,
    )
    return fig


# ── Charts — section 1: Utilisation ──────────────────────────────────────────

st.markdown("### 📊 Resource Utilisation")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**CPU Utilisation (%)**")
    fig_cpu = line_chart(
        df["timestamp"], df["cpu_util_pct"],
        "CPU %", "#3b82f6", "CPU %", [0, 100]
    )
    # Add 85% threshold line
    fig_cpu.add_hline(
        y=85,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="85% threshold",
        annotation_position="top right",
    )
    st.plotly_chart(fig_cpu, use_container_width=True)

with col_right:
    st.markdown("**Bandwidth Utilisation (%)**")
    fig_bw = line_chart(
        df["timestamp"], df["bw_util_pct"],
        "BW %", "#8b5cf6", "BW %", [0, 100]
    )
    fig_bw.add_hline(
        y=90,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="90% threshold",
        annotation_position="top right",
    )
    st.plotly_chart(fig_bw, use_container_width=True)

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("**Memory Utilisation (%)**")
    fig_mem = line_chart(
        df["timestamp"], df["mem_util_pct"],
        "Mem %", "#10b981", "Mem %", [0, 100]
    )
    st.plotly_chart(fig_mem, use_container_width=True)

with col_right2:
    st.markdown("**Active Users**")
    fig_users = line_chart(
        df["timestamp"], df["active_users"],
        "Users", "#f59e0b", "Users"
    )
    st.plotly_chart(fig_users, use_container_width=True)

# ── Charts — section 2: Congestion ───────────────────────────────────────────

st.markdown("### 🔴 Congestion & Queue")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Congestion Flag Timeline**")
    fig_cong = go.Figure(
        go.Scatter(
            x=df["timestamp"],
            y=df["congestion_flag"],
            mode="lines",
            line=dict(color="#ef4444", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
            name="Congested",
        )
    )
    fig_cong.update_layout(
        **PLOT_LAYOUT,
        height=220,
        yaxis_tickvals=[0, 1],
        yaxis_ticktext=["Clear", "Congested"],
        yaxis_range=[-0.1, 1.4],
        yaxis_showgrid=True,
        yaxis_gridcolor="#f1f5f9",
    )
    st.plotly_chart(fig_cong, use_container_width=True)

with col_b:
    st.markdown("**Queue Length Over Time**")
    fig_queue = go.Figure(
        go.Scatter(
            x=df["timestamp"],
            y=df["queue_len"],
            mode="lines",
            line=dict(color="#f97316", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(249, 115, 22, 0.12)",
            name="Queue Len",
        )
    )
    fig_queue.add_hline(
        y=20,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="queue > 20 → congestion",
        annotation_position="top right",
    )
    fig_queue.update_layout(
        **PLOT_LAYOUT,
        height=220,
        yaxis_showgrid=True,
        yaxis_gridcolor="#f1f5f9",
        yaxis_linecolor="#e2e8f0",
        showlegend=False,
    )
    st.plotly_chart(fig_queue, use_container_width=True)

# ── Expander: Raw table ───────────────────────────────────────────────────────

with st.expander("🗄️ Raw data preview (first 200 rows)"):
    display_cols = [
        "timestamp", "slice_id", "slice_type",
        "cpu_util_pct", "mem_util_pct", "bw_util_pct",
        "active_users", "queue_len", "congestion_flag",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].head(200), use_container_width=True)

# ── Expander: Distribution plots ──────────────────────────────────────────────

with st.expander("📐 Distribution of resource metrics"):
    import plotly.express as px

    dcols = st.columns(3)
    for i, (col, label, color) in enumerate([
        ("cpu_util_pct", "CPU %",    "#3b82f6"),
        ("bw_util_pct",  "BW %",     "#8b5cf6"),
        ("mem_util_pct", "Mem %",    "#10b981"),
    ]):
        fig_hist = px.histogram(
            df, x=col, nbins=40, labels={col: label},
            title=f"{label} Distribution",
            color_discrete_sequence=[color],
        )
        fig_hist.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=240,
            margin=dict(l=10, r=10, t=36, b=10),
            showlegend=False,
        )
        dcols[i].plotly_chart(fig_hist, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    "<hr style='border:1px solid #e2e8f0; margin-top:32px'>"
    "<p style='text-align:center; color:#94a3b8; font-size:.8rem'>"
    "6G Network Slicing Analytics Dashboard · Synthetic dataset</p>",
    unsafe_allow_html=True,
)
