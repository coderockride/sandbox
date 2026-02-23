import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------
# App config
# ---------------------------------
st.set_page_config(
    page_title="Commercial Analytics Command Centre",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------
# Formatting helpers
# ---------------------------------
def fmt_eur(value: float) -> str:
    """Format numbers as compact EUR values."""
    if pd.isna(value):
        return "€0"
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"€{value/1_000_000_000:.1f}B"
    if abs_val >= 1_000_000:
        return f"€{value/1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"€{value/1_000:.1f}K"
    return f"€{value:,.0f}"


def fmt_pct(value: float) -> str:
    """Format decimal fractions as percentages."""
    if pd.isna(value):
        return "0.0%"
    return f"{value*100:.1f}%"


def tier_rate(qtr_revenue: float) -> float:
    """Tiered rebate function based on quarterly net revenue."""
    if qtr_revenue < 2_000_000:
        return 0.01
    if qtr_revenue <= 5_000_000:
        return 0.02
    return 0.03


# ---------------------------------
# Synthetic data generation
# ---------------------------------
@st.cache_data
def generate_data(seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic fact table at month x reseller x vendor x category grain."""
    rng = np.random.default_rng(seed)

    months = pd.date_range("2022-01-01", periods=36, freq="MS")

    regions = ["UKI", "France", "DACH", "Nordics", "US", "ANZ"]
    region_weights = {
        "UKI": 0.23,
        "France": 0.17,
        "DACH": 0.20,
        "Nordics": 0.10,
        "US": 0.19,
        "ANZ": 0.11,
    }

    vendors = [
        "Palo Alto",
        "CrowdStrike",
        "Fortinet",
        "Netskope",
        "Rubrik",
        "Proofpoint",
        "SentinelOne",
    ]
    vendor_weights = {
        "Palo Alto": 0.19,
        "CrowdStrike": 0.15,
        "Fortinet": 0.18,
        "Netskope": 0.11,
        "Rubrik": 0.13,
        "Proofpoint": 0.11,
        "SentinelOne": 0.13,
    }

    product_categories = ["Hardware", "Software Licenses", "Support & Maintenance", "Services"]
    cat_base = {
        "Hardware": 0.20,
        "Software Licenses": 0.45,
        "Support & Maintenance": 0.20,
        "Services": 0.15,
    }

    reseller_types = ["MSP", "VAR", "System Integrator", "Telco", "Boutique"]
    type_weights = [0.23, 0.27, 0.22, 0.13, 0.15]

    # 60 realistic reseller names
    reseller_names = [
        "NordSec Solutions GmbH", "BlueGate IT Ltd", "HexaShield SAS", "Alpine Cyber AG",
        "SentraNet Services", "IronWall Partners", "Vertex Secure Ltd", "Aegis Channel Group",
        "NorthBridge Security", "LumenGuard Technologies", "CobaltEdge Systems", "Trident Cyberworks",
        "Aurora Infosec AB", "Keystone Digital Defense", "Meridian Secure Networks", "CrestPoint Cyber",
        "HarborLight Integrations", "Skyline Trust IT", "VectorCore Partners", "Pinnacle Shield Consulting",
        "Polaris Security House", "Atlas Secure Distribution", "CipherTrail Solutions", "NovaSentinel IT",
        "DeepAnchor Tech GmbH", "ShieldCraft SAS", "BlueFjord Security", "Praetorian Link Ltd",
        "Oakline Cyber Services", "GraniteFox Networks", "BridgeWave Technologies", "Argentum Cyber AG",
        "RedMaple Security", "Silverline Data Defense", "Helios Secure Systems", "Zephyr Infosec",
        "FalconPeak Integrators", "Canopy Cyber Partners", "TrueNorth Secure", "QuantaDefend Ltd",
        "Mistral Security France", "DeltaGrid Cyber", "Orion Shieldworks", "Titan Harbor IT",
        "EchoPoint Security", "CrownCastle Integrations", "BastionSphere", "NexaGuard Solutions",
        "WestBridge Cyber", "PrimeArc Security", "ClearPath Infosec", "StoneGate IT Group",
        "NightOwl Digital Security", "Lakeview Cybernetics", "SummitWall Technologies", "Evergreen Secure",
        "CopperSky Cyber", "RapidFort Defense", "SablePoint Security", "Brighton Secure Dynamics",
    ]

    resellers = pd.DataFrame(
        {
            "reseller_id": [f"R{idx:03d}" for idx in range(1, 61)],
            "reseller_name": reseller_names,
            "reseller_type": rng.choice(reseller_types, size=60, p=type_weights),
            "home_region": rng.choice(regions, size=60, p=[region_weights[r] for r in regions]),
        }
    )

    # Concentration: top 8 resellers dominate.
    concentration_weights = np.array([4.5] * 8 + [1.0] * 52)
    concentration_weights = concentration_weights / concentration_weights.sum()
    resellers["size_weight"] = concentration_weights

    # Regional and category trends over time (realism constraints)
    hardware_decline = np.linspace(1.00, 0.72, len(months))
    cloud_increase = np.linspace(0.38, 0.62, len(months))
    time_trend = np.linspace(0.88, 1.20, len(months))

    margin_by_cat = {
        "Hardware": (0.14, 0.20),
        "Software Licenses": (0.24, 0.34),
        "Support & Maintenance": (0.27, 0.38),
        "Services": (0.33, 0.50),
    }

    renewal_rate_by_type = {
        "MSP": 0.90,
        "VAR": 0.84,
        "System Integrator": 0.87,
        "Telco": 0.86,
        "Boutique": 0.82,
    }

    rows = []

    # Build full grain: month x reseller x vendor x category
    for m_idx, month in enumerate(months):
        for _, r in resellers.iterrows():
            for vendor in vendors:
                for cat in product_categories:
                    # Product mix: hardware slowly declines, cloud increases.
                    cat_adj = cat_base[cat]
                    if cat == "Hardware":
                        cat_adj *= hardware_decline[m_idx]
                    elif cat == "Software Licenses":
                        cat_adj *= (1.05 + 0.06 * (m_idx / (len(months) - 1)))
                    elif cat == "Services":
                        cat_adj *= (0.95 + 0.10 * (m_idx / (len(months) - 1)))

                    region_factor = region_weights[r["home_region"]] / np.mean(list(region_weights.values()))
                    vendor_factor = vendor_weights[vendor] / np.mean(list(vendor_weights.values()))
                    concentration = 0.7 + (r["size_weight"] * 60)
                    seasonality = 1 + 0.08 * np.sin((m_idx % 12) / 12 * 2 * np.pi)
                    noise = rng.normal(1.0, 0.18)

                    gross_sales = (
                        22_000
                        * time_trend[m_idx]
                        * cat_adj
                        * region_factor
                        * vendor_factor
                        * concentration
                        * seasonality
                        * max(0.35, noise)
                    )

                    cloud_prob = cloud_increase[m_idx]
                    if cat == "Hardware":
                        cloud_prob *= 0.18
                    elif cat in ["Software Licenses", "Support & Maintenance"]:
                        cloud_prob *= 1.15
                    cloud_flag = rng.random() < min(cloud_prob, 0.98)

                    recurring_prob = 0.05 if cat == "Hardware" else 0.72
                    if cat == "Services":
                        recurring_prob = 0.40
                    recurring_flag = rng.random() < recurring_prob

                    net_share = rng.uniform(0.25, 0.40)
                    net_rev = gross_sales * net_share

                    m_low, m_high = margin_by_cat[cat]
                    target_margin = rng.uniform(m_low, m_high)

                    # Margin leakage pockets and low margin anomalies.
                    leakage = 0.0
                    if rng.random() < 0.03:
                        leakage += rng.uniform(0.04, 0.10)
                    if r["home_region"] in ["France", "ANZ"] and cat == "Hardware" and rng.random() < 0.10:
                        leakage += rng.uniform(0.02, 0.06)

                    realized_margin = max(0.02, target_margin - leakage)
                    cogs = net_rev * (1 - realized_margin)
                    gross_margin = net_rev - cogs

                    renewals_due = 0.0
                    renewals_won = 0.0
                    if recurring_flag:
                        renewals_due = net_rev * rng.uniform(0.35, 0.80)
                        base_rr = renewal_rate_by_type[r["reseller_type"]]
                        rr = np.clip(rng.normal(base_rr, 0.06), 0.55, 0.99)
                        if rng.random() < 0.05:
                            rr *= rng.uniform(0.75, 0.92)
                        renewals_won = min(renewals_due, renewals_due * rr)

                    rebate_rate = np.clip(0.01 + 0.015 * (gross_margin / max(net_rev, 1)), 0.008, 0.035)
                    rebates_earned = net_rev * rebate_rate * rng.uniform(0.85, 1.15)

                    invoices_count = int(max(1, round(gross_sales / rng.uniform(12_000, 55_000))))
                    credits_count = 0
                    notes = ""

                    # Occasional credits and explicit anomalies.
                    if rng.random() < 0.03:
                        credits_count = int(rng.integers(1, 4))
                        notes = "credit issued"
                    if gross_margin / max(net_rev, 1) < 0.08 and rng.random() < 0.5:
                        notes = "pricing exception"
                    if recurring_flag and renewals_due > 0 and (renewals_won / renewals_due) < 0.80 and rng.random() < 0.6:
                        notes = "late renewal"

                    rows.append(
                        {
                            "month": month,
                            "region": r["home_region"],
                            "reseller_id": r["reseller_id"],
                            "reseller_name": r["reseller_name"],
                            "reseller_type": r["reseller_type"],
                            "vendor": vendor,
                            "product_category": cat,
                            "cloud_flag": bool(cloud_flag),
                            "recurring_flag": bool(recurring_flag),
                            "gross_sales_eur": gross_sales,
                            "net_revenue_eur": net_rev,
                            "cogs_eur": cogs,
                            "gross_margin_eur": gross_margin,
                            "gross_margin_pct": gross_margin / max(net_rev, 1),
                            "renewals_due_eur": renewals_due,
                            "renewals_won_eur": renewals_won,
                            "rebates_earned_eur": rebates_earned,
                            "invoices_count": invoices_count,
                            "credits_count": credits_count,
                            "notes": notes,
                        }
                    )

    df = pd.DataFrame(rows)

    # Final cleanup and friendly month label.
    df["month_str"] = df["month"].dt.strftime("%Y-%m")
    df["gross_margin_pct"] = np.where(df["net_revenue_eur"] > 0, df["gross_margin_eur"] / df["net_revenue_eur"], 0)

    return df


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp],
    regions: list[str],
    vendors: list[str],
    categories: list[str],
) -> pd.DataFrame:
    """Apply global sidebar filters to the fact table."""
    start_date, end_date = date_range
    mask = (
        (df["month"] >= pd.to_datetime(start_date))
        & (df["month"] <= pd.to_datetime(end_date))
        & (df["region"].isin(regions))
        & (df["vendor"].isin(vendors))
        & (df["product_category"].isin(categories))
    )
    return df.loc[mask].copy()


# ---------------------------------
# Load data and global filters
# ---------------------------------
df = generate_data(seed=42)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Executive Overview", "Partner (Reseller) Intelligence", "Vendor & Rebate Lens"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

min_month = df["month"].min().date()
max_month = df["month"].max().date()
selected_dates = st.sidebar.date_input(
    "Date range",
    value=(min_month, max_month),
    min_value=min_month,
    max_value=max_month,
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    date_start, date_end = selected_dates
else:
    date_start, date_end = min_month, max_month

a_regions = sorted(df["region"].unique().tolist())
a_vendors = sorted(df["vendor"].unique().tolist())
a_categories = sorted(df["product_category"].unique().tolist())

sel_regions = st.sidebar.multiselect("Region", a_regions, default=a_regions)
sel_vendors = st.sidebar.multiselect("Vendor", a_vendors, default=a_vendors)
sel_categories = st.sidebar.multiselect("Product category", a_categories, default=a_categories)

if not sel_regions:
    sel_regions = a_regions
if not sel_vendors:
    sel_vendors = a_vendors
if not sel_categories:
    sel_categories = a_categories

fdf = apply_filters(
    df,
    (pd.to_datetime(date_start), pd.to_datetime(date_end)),
    sel_regions,
    sel_vendors,
    sel_categories,
)

if fdf.empty:
    st.warning("No data matches the current filter selection. Try broadening the filters.")
    st.stop()


# ---------------------------------
# Page 1: Executive Overview
# ---------------------------------
if page == "Executive Overview":
    st.title("Commercial Analytics Command Centre")
    st.caption("Executive Overview")

    gross_sales = fdf["gross_sales_eur"].sum()
    net_revenue = fdf["net_revenue_eur"].sum()
    gross_margin = fdf["gross_margin_eur"].sum()
    margin_pct = gross_margin / net_revenue if net_revenue > 0 else 0
    cloud_pct = (fdf.loc[fdf["cloud_flag"], "net_revenue_eur"].sum() / net_revenue) if net_revenue > 0 else 0
    recurring_pct = (fdf.loc[fdf["recurring_flag"], "net_revenue_eur"].sum() / net_revenue) if net_revenue > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Gross Sales", fmt_eur(gross_sales))
    c2.metric("Net Revenue", fmt_eur(net_revenue))
    c3.metric("Gross Margin €", fmt_eur(gross_margin))
    c4.metric("Margin %", fmt_pct(margin_pct))
    c5.metric("Cloud %", fmt_pct(cloud_pct))
    c6.metric("Recurring %", fmt_pct(recurring_pct))

    monthly = (
        fdf.groupby("month", as_index=False)[["gross_sales_eur", "net_revenue_eur"]]
        .sum()
        .sort_values("month")
    )
    fig_line = px.line(
        monthly,
        x="month",
        y=["gross_sales_eur", "net_revenue_eur"],
        title="Gross Sales vs Net Revenue Over Time",
        labels={"value": "EUR", "month": "Month", "variable": "Metric"},
    )
    fig_line.update_layout(legend_title_text="")
    st.plotly_chart(fig_line, use_container_width=True)

    mix = (
        fdf.groupby(["month", "product_category"], as_index=False)["net_revenue_eur"]
        .sum()
        .sort_values("month")
    )
    fig_area = px.area(
        mix,
        x="month",
        y="net_revenue_eur",
        color="product_category",
        title="Net Revenue Mix by Product Category",
        labels={"net_revenue_eur": "Net Revenue (EUR)", "month": "Month", "product_category": "Category"},
    )
    st.plotly_chart(fig_area, use_container_width=True)

    top10 = (
        fdf.groupby(["reseller_name"], as_index=False)
        .agg(
            net_revenue_eur=("net_revenue_eur", "sum"),
            gross_margin_eur=("gross_margin_eur", "sum"),
            cloud_rev=("net_revenue_eur", lambda s: 0),
        )
        .sort_values("net_revenue_eur", ascending=False)
        .head(10)
    )

    cloud_by_reseller = (
        fdf[fdf["cloud_flag"]]
        .groupby("reseller_name", as_index=False)["net_revenue_eur"]
        .sum()
        .rename(columns={"net_revenue_eur": "cloud_rev"})
    )
    top10 = top10.drop(columns=["cloud_rev"]).merge(cloud_by_reseller, on="reseller_name", how="left")
    top10["cloud_rev"] = top10["cloud_rev"].fillna(0)
    top10["margin_pct"] = np.where(top10["net_revenue_eur"] > 0, top10["gross_margin_eur"] / top10["net_revenue_eur"], 0)
    top10["cloud_pct"] = np.where(top10["net_revenue_eur"] > 0, top10["cloud_rev"] / top10["net_revenue_eur"], 0)

    disp_top10 = top10[["reseller_name", "net_revenue_eur", "margin_pct", "cloud_pct"]].copy()
    disp_top10["net_revenue_eur"] = disp_top10["net_revenue_eur"].map(fmt_eur)
    disp_top10["margin_pct"] = disp_top10["margin_pct"].map(fmt_pct)
    disp_top10["cloud_pct"] = disp_top10["cloud_pct"].map(fmt_pct)

    st.subheader("Top 10 Resellers by Net Revenue")
    st.dataframe(disp_top10, use_container_width=True, hide_index=True)

    st.caption("Synthetic notional data for demonstration only.")


# ---------------------------------
# Page 2: Partner Intelligence
# ---------------------------------
elif page == "Partner (Reseller) Intelligence":
    st.title("Partner (Reseller) Intelligence")

    reseller_rank = (
        fdf.groupby(["reseller_id", "reseller_name"], as_index=False)["net_revenue_eur"]
        .sum()
        .sort_values("net_revenue_eur", ascending=False)
    )
    default_reseller = reseller_rank.iloc[0]["reseller_id"]
    reseller_map = dict(zip(reseller_rank["reseller_name"], reseller_rank["reseller_id"]))
    name_options = reseller_rank["reseller_name"].tolist()

    default_name = reseller_rank.iloc[0]["reseller_name"]
    selected_name = st.selectbox("Select reseller", options=name_options, index=name_options.index(default_name))
    selected_id = reseller_map[selected_name]

    rdf = fdf[fdf["reseller_id"] == selected_id].copy()

    profile_type = rdf["reseller_type"].mode().iloc[0] if not rdf.empty else "N/A"
    profile_regions = ", ".join(sorted(rdf["region"].unique().tolist()))
    top3_vendors = (
        rdf.groupby("vendor", as_index=False)["net_revenue_eur"].sum().sort_values("net_revenue_eur", ascending=False).head(3)["vendor"].tolist()
    )

    latest_month = fdf["month"].max()
    l12_start = latest_month - pd.DateOffset(months=11)
    prev12_start = latest_month - pd.DateOffset(months=23)
    prev12_end = latest_month - pd.DateOffset(months=12)

    l12 = rdf[(rdf["month"] >= l12_start) & (rdf["month"] <= latest_month)]["net_revenue_eur"].sum()
    p12 = rdf[(rdf["month"] >= prev12_start) & (rdf["month"] <= prev12_end)]["net_revenue_eur"].sum()
    l12_growth = ((l12 / p12) - 1) if p12 > 0 else np.nan

    st.markdown(
        f"""
        **Reseller Type:** {profile_type}  
        **Active Regions:** {profile_regions if profile_regions else 'N/A'}  
        **Top 3 Vendors:** {', '.join(top3_vendors) if top3_vendors else 'N/A'}  
        **Last 12M Growth:** {fmt_pct(l12_growth) if pd.notna(l12_growth) else 'N/A'}
        """
    )

    # Opportunity flags
    st.subheader("Opportunity Flags")
    flag_cols = st.columns(3)

    # 1) Under-penetrated vendor portfolio
    vendor_count_selected = rdf["vendor"].nunique()
    peer = fdf[fdf["reseller_type"] == profile_type]
    peer_vendor_counts = peer.groupby("reseller_id")["vendor"].nunique()
    peer_median = peer_vendor_counts.median() if not peer_vendor_counts.empty else vendor_count_selected
    under_penetrated = vendor_count_selected < peer_median
    flag_cols[0].metric(
        "Under-penetrated portfolio",
        "Yes" if under_penetrated else "No",
        delta=f"Vendors {vendor_count_selected} vs peer median {peer_median:.0f}",
    )

    # 2) Margin pressure
    reseller_margin = rdf["gross_margin_eur"].sum() / rdf["net_revenue_eur"].sum() if rdf["net_revenue_eur"].sum() > 0 else 0
    region_medians = []
    for reg in rdf["region"].unique():
        reg_df = fdf[fdf["region"] == reg]
        reg_margin = reg_df["gross_margin_eur"].sum() / reg_df["net_revenue_eur"].sum() if reg_df["net_revenue_eur"].sum() > 0 else 0
        region_medians.append(reg_margin)
    region_median_margin = float(np.median(region_medians)) if region_medians else reseller_margin
    threshold = 0.03
    margin_pressure = reseller_margin < (region_median_margin - threshold)
    flag_cols[1].metric(
        "Margin pressure",
        "Yes" if margin_pressure else "No",
        delta=f"{fmt_pct(reseller_margin)} vs region median {fmt_pct(region_median_margin)}",
    )

    # 3) Renewal risk
    recent6_start = latest_month - pd.DateOffset(months=5)
    recent6 = rdf[(rdf["month"] >= recent6_start) & (rdf["month"] <= latest_month)]
    due6 = recent6["renewals_due_eur"].sum()
    won6 = recent6["renewals_won_eur"].sum()
    renewal_rate_6m = won6 / due6 if due6 > 0 else np.nan
    renewal_risk = (pd.notna(renewal_rate_6m)) and (renewal_rate_6m < 0.85)
    flag_cols[2].metric(
        "Renewal risk",
        "Yes" if renewal_risk else "No",
        delta=f"6M renewal rate {fmt_pct(renewal_rate_6m) if pd.notna(renewal_rate_6m) else 'N/A'}",
    )

    c1, c2 = st.columns(2)

    trend = rdf.groupby("month", as_index=False)["net_revenue_eur"].sum().sort_values("month")
    fig_reseller_trend = px.line(
        trend,
        x="month",
        y="net_revenue_eur",
        title="Monthly Net Revenue Trend",
        labels={"month": "Month", "net_revenue_eur": "Net Revenue (EUR)"},
    )
    c1.plotly_chart(fig_reseller_trend, use_container_width=True)

    last12 = rdf[(rdf["month"] >= l12_start) & (rdf["month"] <= latest_month)]
    vmix = last12.groupby("vendor", as_index=False)["net_revenue_eur"].sum().sort_values("net_revenue_eur", ascending=False)
    fig_vendor_mix = px.bar(
        vmix,
        x="vendor",
        y="net_revenue_eur",
        title="Vendor Mix (Last 12 Months)",
        labels={"vendor": "Vendor", "net_revenue_eur": "Net Revenue (EUR)"},
    )
    c2.plotly_chart(fig_vendor_mix, use_container_width=True)

    renew = rdf.groupby("month", as_index=False)[["renewals_due_eur", "renewals_won_eur"]].sum().sort_values("month")
    renew["renewal_rate"] = np.where(renew["renewals_due_eur"] > 0, renew["renewals_won_eur"] / renew["renewals_due_eur"], np.nan)
    fig_renew = px.line(
        renew,
        x="month",
        y="renewal_rate",
        title="Renewal Rate Trend",
        labels={"month": "Month", "renewal_rate": "Renewal Rate"},
    )
    fig_renew.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig_renew, use_container_width=True)


# ---------------------------------
# Page 3: Vendor & Rebate Lens
# ---------------------------------
else:
    st.title("Vendor & Rebate Lens")

    vendor_options = sorted(fdf["vendor"].unique().tolist())
    selected_vendor = st.selectbox("Select vendor", vendor_options)

    vdf = fdf[fdf["vendor"] == selected_vendor].copy()

    if vdf.empty:
        st.info("No rows available for this vendor under current filters.")
        st.stop()

    v_gross = vdf["gross_sales_eur"].sum()
    v_net = vdf["net_revenue_eur"].sum()
    v_margin = vdf["gross_margin_eur"].sum() / v_net if v_net > 0 else 0
    v_rebates = vdf["rebates_earned_eur"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gross Sales", fmt_eur(v_gross))
    k2.metric("Net Revenue", fmt_eur(v_net))
    k3.metric("Margin %", fmt_pct(v_margin))
    k4.metric("Rebates Earned", fmt_eur(v_rebates))

    rebate_ts = vdf.groupby("month", as_index=False)["rebates_earned_eur"].sum().sort_values("month")
    fig_rebate = px.bar(
        rebate_ts,
        x="month",
        y="rebates_earned_eur",
        title="Rebates Earned Over Time",
        labels={"month": "Month", "rebates_earned_eur": "Rebates (EUR)"},
    )
    st.plotly_chart(fig_rebate, use_container_width=True)

    top_vendor_resellers = (
        vdf.groupby("reseller_name", as_index=False)
        .agg(
            net_revenue_eur=("net_revenue_eur", "sum"),
            gross_margin_eur=("gross_margin_eur", "sum"),
            renewals_due_eur=("renewals_due_eur", "sum"),
            renewals_won_eur=("renewals_won_eur", "sum"),
        )
        .sort_values("net_revenue_eur", ascending=False)
        .head(12)
    )
    top_vendor_resellers["margin_pct"] = np.where(
        top_vendor_resellers["net_revenue_eur"] > 0,
        top_vendor_resellers["gross_margin_eur"] / top_vendor_resellers["net_revenue_eur"],
        0,
    )
    top_vendor_resellers["renewal_rate"] = np.where(
        top_vendor_resellers["renewals_due_eur"] > 0,
        top_vendor_resellers["renewals_won_eur"] / top_vendor_resellers["renewals_due_eur"],
        np.nan,
    )

    display_vendor_tbl = top_vendor_resellers[["reseller_name", "net_revenue_eur", "margin_pct", "renewal_rate"]].copy()
    display_vendor_tbl["net_revenue_eur"] = display_vendor_tbl["net_revenue_eur"].map(fmt_eur)
    display_vendor_tbl["margin_pct"] = display_vendor_tbl["margin_pct"].map(fmt_pct)
    display_vendor_tbl["renewal_rate"] = display_vendor_tbl["renewal_rate"].map(lambda x: fmt_pct(x) if pd.notna(x) else "N/A")

    st.subheader("Top Resellers for Selected Vendor")
    st.dataframe(display_vendor_tbl, use_container_width=True, hide_index=True)

    st.subheader("Rebate Tier Simulator")

    latest_month = vdf["month"].max()
    qtr_start = latest_month - pd.DateOffset(months=2)
    current_qtr_net = vdf[(vdf["month"] >= qtr_start) & (vdf["month"] <= latest_month)]["net_revenue_eur"].sum()

    sensible_max = max(500_000.0, current_qtr_net * 1.5)
    addl = st.slider(
        "Additional net revenue next quarter (€)",
        min_value=0.0,
        max_value=float(sensible_max),
        value=float(sensible_max * 0.2),
        step=float(max(10_000.0, sensible_max / 100)),
    )

    curr_rate = tier_rate(current_qtr_net)
    proj_qtr_net = current_qtr_net + addl
    proj_rate = tier_rate(proj_qtr_net)

    current_rebate = current_qtr_net * curr_rate
    projected_rebate = proj_qtr_net * proj_rate
    incremental_rebate = projected_rebate - current_rebate

    t1, t2, t3 = st.columns(3)
    t1.metric("Current Tier", f"{int(curr_rate*100)}%", delta=f"Quarterly net {fmt_eur(current_qtr_net)}")
    t2.metric("Projected Tier", f"{int(proj_rate*100)}%", delta=f"Projected net {fmt_eur(proj_qtr_net)}")
    t3.metric("Incremental Rebate €", fmt_eur(incremental_rebate))

    st.caption("Synthetic notional data for demonstration only.")
