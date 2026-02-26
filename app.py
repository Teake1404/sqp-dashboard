"""
SQP (Search Query Performance) Dashboard
Demonstrates Brand Analytics automation for Amazon sellers.
Built with Streamlit — deployable to share.streamlit.io
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import random
import io

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SQP Dashboard · Brand Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Sample data generator ─────────────────────────────────────────────────────

def generate_sqp_data():
    random.seed(42)

    clients = {
        "ZenFit (Protein Supplements)": {
            "queries": [
                "whey protein powder",
                "protein powder for women",
                "chocolate whey protein",
                "best protein powder uk",
                "whey protein 1kg",
                "plant based protein powder",
                "protein supplement gym",
                "low calorie protein shake",
                "isolate protein powder",
                "post workout protein",
            ],
            "base_impressions": [95000, 72000, 58000, 51000, 44000, 38000, 33000, 27000, 22000, 18000],
            "base_share": [0.08, 0.14, 0.11, 0.06, 0.09, 0.18, 0.12, 0.07, 0.05, 0.10],
        },
        "VitaCore (Vitamins & Wellness)": {
            "queries": [
                "vitamin d3 supplement",
                "multivitamin for men",
                "omega 3 fish oil",
                "vitamin c tablets",
                "b12 supplement",
                "zinc and vitamin c",
                "immunity booster tablets",
                "magnesium supplement",
                "iron tablets women",
                "biotin tablets hair growth",
            ],
            "base_impressions": [88000, 66000, 54000, 47000, 41000, 35000, 29000, 24000, 20000, 15000],
            "base_share": [0.12, 0.09, 0.15, 0.11, 0.07, 0.13, 0.08, 0.10, 0.06, 0.16],
        },
    }

    weeks = []
    today = date.today()
    days_back = (today.weekday() + 1) % 7
    last_sunday = today - timedelta(days=days_back)
    for w in range(8, 0, -1):
        week_end = last_sunday - timedelta(weeks=w - 1)
        week_start = week_end - timedelta(days=6)
        weeks.append((week_start, week_end))

    rows = []
    for client_name, cfg in clients.items():
        for i, query in enumerate(cfg["queries"]):
            base_imp = cfg["base_impressions"][i]
            base_share = cfg["base_share"][i]

            trend_dir = 1 if i % 3 != 2 else -1
            trend_strength = random.uniform(0.01, 0.04)

            for w_idx, (wk_start, wk_end) in enumerate(weeks):
                noise = random.uniform(-0.02, 0.02)
                share = max(0.01, min(0.50,
                    base_share + trend_dir * trend_strength * w_idx + noise
                ))
                imp = int(base_imp * random.uniform(0.85, 1.15))
                clicks = int(imp * share * random.uniform(0.8, 1.2))
                cart_adds = int(clicks * random.uniform(0.25, 0.45))
                purchases = int(cart_adds * random.uniform(0.55, 0.75))

                rows.append({
                    "client": client_name,
                    "week_start": wk_start,
                    "week_end": wk_end,
                    "week_label": wk_start.strftime("%b %d"),
                    "search_query": query,
                    "impressions": imp,
                    "clicks": clicks,
                    "cart_adds": cart_adds,
                    "purchases": purchases,
                    "purchase_share": round(share, 4),
                    "ctr": round(clicks / imp, 4) if imp > 0 else 0,
                    "cvr": round(purchases / clicks, 4) if clicks > 0 else 0,
                })

    df = pd.DataFrame(rows)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = pd.to_datetime(df["week_end"])
    return df


# ── CSV parser (for uploaded files) ──────────────────────────────────────────

def parse_uploaded_sqp(file) -> pd.DataFrame:
    """
    Parses an SQP CSV exported from Amazon Seller Central → Brand Analytics.
    Expected columns (Amazon's actual export format):
      Search Query, Search Query Score, Impressions, Clicks,
      Cart Adds, Purchases, Brand Share
    """
    try:
        df = pd.read_csv(file)
    except Exception:
        return None

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    col_map = {
        "search_query":         "search_query",
        "search_query_score":   "impressions",      # fallback if no impressions col
        "impressions":          "impressions",
        "clicks":               "clicks",
        "cart_adds":            "cart_adds",
        "purchases":            "purchases",
        "brand_share":          "purchase_share",
        "purchase_share":       "purchase_share",
    }

    renamed = {}
    for src, dst in col_map.items():
        if src in df.columns:
            renamed[src] = dst
    df = df.rename(columns=renamed)

    required = ["search_query", "impressions", "clicks", "purchases"]
    if not all(c in df.columns for c in required):
        return None

    for col in ["impressions", "clicks", "purchases"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)

    if "purchase_share" not in df.columns:
        df["purchase_share"] = 0.0
    else:
        df["purchase_share"] = (
            df["purchase_share"].astype(str).str.replace("%", "").str.replace(",", "")
        )
        df["purchase_share"] = pd.to_numeric(df["purchase_share"], errors="coerce").fillna(0)
        if df["purchase_share"].max() > 1:
            df["purchase_share"] = df["purchase_share"] / 100

    if "cart_adds" not in df.columns:
        df["cart_adds"] = (df["clicks"] * 0.35).astype(int)

    df["ctr"] = (df["clicks"] / df["impressions"].replace(0, 1)).round(4)
    df["cvr"] = (df["purchases"] / df["clicks"].replace(0, 1)).round(4)

    # Add week/client columns for compatibility with dashboard
    today = date.today()
    days_back = (today.weekday() + 1) % 7
    week_start = today - timedelta(days=days_back + 6)
    df["client"]     = "Your Brand"
    df["week_start"] = pd.to_datetime(week_start)
    df["week_end"]   = pd.to_datetime(today - timedelta(days=days_back))
    df["week_label"] = pd.to_datetime(week_start).strftime("%b %d")

    return df[["client", "week_start", "week_end", "week_label",
               "search_query", "impressions", "clicks", "cart_adds",
               "purchases", "purchase_share", "ctr", "cvr"]]


# ── Shared analytics functions ─────────────────────────────────────────────────

def compute_opportunity_score(df_week):
    max_imp = df_week["impressions"].max() or 1
    df = df_week.copy()
    df["imp_norm"] = df["impressions"] / max_imp
    df["opportunity_score"] = (df["imp_norm"] * (1 - df["purchase_share"]) * 100).round(1)
    return df


def compute_wow_delta(df, current_week, prev_week):
    cur  = df[df["week_start"] == current_week][["search_query", "purchase_share"]].copy()
    prev = df[df["week_start"] == prev_week][["search_query", "purchase_share"]].copy()
    merged = cur.merge(prev, on="search_query", suffixes=("_cur", "_prev"))
    merged["share_delta"] = (merged["purchase_share_cur"] - merged["purchase_share_prev"]).round(4)
    merged["delta_pct"]   = ((merged["share_delta"] / merged["purchase_share_prev"].replace(0, 0.0001)) * 100).round(1)
    return merged


@st.cache_data
def load_demo_data():
    return generate_sqp_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 SQP Dashboard")
    st.caption("Brand Analytics · Search Query Performance")
    st.divider()

    data_mode = st.radio(
        "Data source",
        ["📂 Upload your SQP CSV", "🎯 View demo data"],
        index=1,
    )

    uploaded_df = None
    if data_mode == "📂 Upload your SQP CSV":
        st.markdown("**How to export your SQP CSV:**")
        st.caption(
            "1. Seller Central → Brand Analytics\n"
            "2. Search Query Performance\n"
            "3. Select ASIN + week range\n"
            "4. Click Generate Download"
        )
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            uploaded_df = parse_uploaded_sqp(uploaded_file)
            if uploaded_df is None:
                st.error("Could not parse this CSV. Make sure it's an Amazon SQP export.")
            else:
                st.success(f"✅ {len(uploaded_df)} keywords loaded")

    st.divider()

    if data_mode == "🎯 View demo data" or uploaded_df is None:
        demo_df   = load_demo_data()
        all_clients = demo_df["client"].unique().tolist()
        selected_client = st.selectbox("Demo client", all_clients)
        df_client = demo_df[demo_df["client"] == selected_client]
        all_weeks = sorted(df_client["week_start"].unique())
        week_labels = [w.strftime("%b %d, %Y") for w in all_weeks]
        selected_week_idx = st.selectbox(
            "Week",
            range(len(all_weeks)),
            index=len(all_weeks) - 1,
            format_func=lambda i: week_labels[i],
        )
        selected_week = all_weeks[selected_week_idx]
        df_week = df_client[df_client["week_start"] == selected_week].copy()
        brand_label = selected_client
        is_single_week = False
    else:
        df_client = uploaded_df
        df_week   = uploaded_df.copy()
        selected_week = uploaded_df["week_start"].iloc[0]
        selected_week_idx = 0
        brand_label = "Your Brand"
        is_single_week = True

    st.divider()
    st.caption("SP-API · Brand Analytics\nGET_BRAND_ANALYTICS_SEARCH_QUERY_PERFORMANCE_REPORT")
    st.caption("Auto-pulled every Sunday · Built by Shuqing")

df_week = compute_opportunity_score(df_week)

# ── Header ────────────────────────────────────────────────────────────────────

st.title(f"📊 SQP Dashboard — {brand_label.split('(')[0].strip()}")
st.caption(
    f"Week of {pd.Timestamp(selected_week).strftime('%B %d, %Y')}  ·  "
    f"{len(df_week)} search queries  ·  UK marketplace"
)

# ── Lead magnet CTA (shown when using uploaded data) ──────────────────────────

if data_mode == "📂 Upload your SQP CSV" and uploaded_df is not None:
    st.info(
        "**Want this report automatically every week — for every ASIN?**  \n"
        "I pull your Brand Analytics data via SP-API, analyse trends, and surface keyword opportunities. "
        "No more manual CSV downloads.  \n"
        "👉 [Book a free 15-min discovery chat](https://calendly.com/shuqingke1404/15-minute-dscovery-chat)",
        icon="🚀",
    )

# ── KPI cards ─────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

total_imp    = df_week["impressions"].sum()
total_clicks = df_week["clicks"].sum()
total_purch  = df_week["purchases"].sum()
avg_share    = df_week["purchase_share"].mean()
avg_cvr      = (total_purch / total_clicks) if total_clicks > 0 else 0

col1.metric("Total Impressions", f"{total_imp:,.0f}")
col2.metric("Total Clicks",      f"{total_clicks:,.0f}")
col3.metric("Purchases",         f"{total_purch:,.0f}")
col4.metric("Avg Purchase Share", f"{avg_share*100:.1f}%")
col5.metric("Avg CVR",           f"{avg_cvr*100:.1f}%")

st.divider()

# ── Charts row 1 ──────────────────────────────────────────────────────────────

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.subheader("🎯 Keyword Opportunity Map")
    st.caption("Large bubble = high search volume · Low purchase share = untapped opportunity")

    fig_bubble = px.scatter(
        df_week.sort_values("opportunity_score", ascending=False),
        x="purchase_share",
        y="impressions",
        size="opportunity_score",
        color="cvr",
        hover_name="search_query",
        hover_data={
            "impressions":     ":,",
            "purchase_share":  ":.1%",
            "cvr":             ":.1%",
            "opportunity_score": ":.1f",
        },
        color_continuous_scale="RdYlGn",
        size_max=60,
        labels={
            "purchase_share": "Your Purchase Share",
            "impressions":    "Total Impressions",
            "cvr":            "CVR",
        },
    )
    fig_bubble.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="CVR", tickformat=".0%"),
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

with right:
    if not is_single_week:
        st.subheader("📈 Purchase Share Trend")
        st.caption("Top 5 keywords — weekly purchase share over 8 weeks")

        top5 = df_week.nlargest(5, "purchases")["search_query"].tolist()
        df_trend = df_client[df_client["search_query"].isin(top5)].sort_values("week_start")

        fig_trend = px.line(
            df_trend,
            x="week_label",
            y="purchase_share",
            color="search_query",
            markers=True,
            labels={
                "week_label":     "Week",
                "purchase_share": "Purchase Share",
                "search_query":   "Query",
            },
        )
        fig_trend.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat=".0%",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.55,
                font=dict(size=10),
            ),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.subheader("🏆 Top Keywords by Opportunity")
        st.caption("Upload multiple weeks to see trends over time")

        top_opps = df_week.nlargest(8, "opportunity_score")[
            ["search_query", "impressions", "purchase_share", "cvr", "opportunity_score"]
        ]
        fig_bar = px.bar(
            top_opps.sort_values("opportunity_score"),
            x="opportunity_score",
            y="search_query",
            orientation="h",
            color="purchase_share",
            color_continuous_scale="RdYlGn_r",
            labels={
                "opportunity_score": "Opportunity Score",
                "search_query":      "",
                "purchase_share":    "Your Share",
            },
        )
        fig_bar.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Charts row 2 ──────────────────────────────────────────────────────────────

left2, right2 = st.columns([1, 1], gap="large")

with left2:
    st.subheader("🔺🔻 Week-over-Week Share Change")
    st.caption("Winners gaining share · Losers losing ground")

    if not is_single_week and selected_week_idx > 0:
        prev_week = all_weeks[selected_week_idx - 1]
        wow = compute_wow_delta(df_client, selected_week, prev_week)
        wow_sorted = wow.sort_values("delta_pct", ascending=False)

        display = wow_sorted[["search_query", "purchase_share_cur", "purchase_share_prev", "delta_pct"]].copy()
        display.columns = ["Search Query", "This Week", "Last Week", "Δ%"]
        display["This Week"] = display["This Week"].map("{:.1%}".format)
        display["Last Week"] = display["Last Week"].map("{:.1%}".format)
        display["Δ%"]        = display["Δ%"].map("{:+.1f}%".format)

        st.dataframe(display, use_container_width=True, height=320, hide_index=True)
    else:
        st.info("Upload multiple weeks of SQP CSVs to see week-over-week changes.")

with right2:
    st.subheader("🔽 Conversion Funnel")
    st.caption("Aggregated across all queries for selected week")

    funnel_data = pd.DataFrame({
        "Stage": ["Impressions", "Clicks", "Cart Adds", "Purchases"],
        "Count": [
            df_week["impressions"].sum(),
            df_week["clicks"].sum(),
            df_week["cart_adds"].sum(),
            df_week["purchases"].sum(),
        ],
    })

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_data["Stage"],
        x=funnel_data["Count"],
        textinfo="value+percent initial",
        marker=dict(color=["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"]),
    ))
    fig_funnel.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# ── Full keyword table ────────────────────────────────────────────────────────

st.subheader("📋 Full Keyword Table")

col_filter, col_sort = st.columns([2, 1])
with col_filter:
    search = st.text_input("Filter keywords", placeholder="e.g. protein, whey...")
with col_sort:
    sort_by = st.selectbox("Sort by", ["opportunity_score", "impressions", "purchases", "purchase_share", "cvr"])

display_df = df_week.copy()
if search:
    display_df = display_df[display_df["search_query"].str.contains(search, case=False)]

display_df = display_df.sort_values(sort_by, ascending=False)

table_cols = ["search_query", "impressions", "clicks", "cart_adds", "purchases",
              "purchase_share", "ctr", "cvr", "opportunity_score"]
display_df_show = display_df[table_cols].copy()
display_df_show.columns = [
    "Search Query", "Impressions", "Clicks", "Cart Adds", "Purchases",
    "Purchase Share", "CTR", "CVR", "Opportunity Score",
]

st.dataframe(
    display_df_show.style.format({
        "Impressions":      "{:,.0f}",
        "Clicks":           "{:,.0f}",
        "Cart Adds":        "{:,.0f}",
        "Purchases":        "{:,.0f}",
        "Purchase Share":   "{:.1%}",
        "CTR":              "{:.1%}",
        "CVR":              "{:.1%}",
        "Opportunity Score": "{:.1f}",
    }).background_gradient(subset=["Opportunity Score"], cmap="YlOrRd"),
    use_container_width=True,
    hide_index=True,
    height=400,
)

# ── Bottom CTA ─────────────────────────────────────────────────────────────────

st.divider()
cta_left, cta_right = st.columns([2, 1])
with cta_left:
    st.markdown(
        "**Want this automatically every week for your brand?**  \n"
        "I connect to your Amazon account via SP-API, pull SQP data weekly, "
        "and surface keyword gaps before your competitors spot them."
    )
with cta_right:
    st.link_button("📅 Book a free 15-min chat", "https://calendly.com/shuqingke1404/15-minute-dscovery-chat", use_container_width=True)

st.caption(
    "Built with Amazon SP-API · Brand Analytics SQP Report · "
    "Automated weekly pull · shuqingke1404@gmail.com"
)
