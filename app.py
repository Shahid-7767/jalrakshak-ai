"""JalRakshak AI - Intelligent Rural Water Management System (core version)."""
import datetime as dt
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

DB_PATH = "jalrakshak.db"
PERSONS_PER_HOUSEHOLD = 5
ESSENTIAL_LPCD = 20          # litres per person per day for drinking and cooking
NON_ESSENTIAL_NORM = 60      # litres per household per day for washing, bathing etc.
RAIN_RUNOFF = 0.0004         # fraction of tank capacity gained per mm of rain (demo assumption)

st.set_page_config(page_title="JalRakshak AI", page_icon="💧", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: "Segoe UI", Roboto, Arial, sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }
    h1, h2, h3 { color: #0b3c6e; }
    [data-testid="stSidebar"] { background: #0b2545; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #e6f0ff !important; }
    div[data-testid="stMetric"] {
        background: #eef6ff; border: 1px solid #cfe6ff; border-radius: 12px;
        padding: 14px 16px; box-shadow: 0 1px 3px rgba(11, 60, 110, 0.08);
    }
    div[data-testid="stMetricValue"] { color: #0b3c6e; }
    .jal-banner {
        background: linear-gradient(90deg, #0b3c6e, #1e88e5); color: white;
        padding: 18px 24px; border-radius: 14px; margin-bottom: 18px;
    }
    .jal-banner h2 { color: white; margin: 0; }
    .jal-banner p { margin: 4px 0 0 0; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- database
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS villages(
            id INTEGER PRIMARY KEY, name TEXT UNIQUE,
            capacity_l REAL, households INTEGER);
        CREATE TABLE IF NOT EXISTS daily(
            id INTEGER PRIMARY KEY, village_id INTEGER, day TEXT,
            level_l REAL, supply_l REAL, consumption_l REAL, rainfall_mm REAL);
        CREATE TABLE IF NOT EXISTS alerts(
            id INTEGER PRIMARY KEY, village_id INTEGER, created TEXT,
            severity TEXT, message TEXT, resolved INTEGER DEFAULT 0);
        """
    )
    if conn.execute("SELECT COUNT(*) FROM villages").fetchone()[0] == 0:
        seed_demo_data(conn)
    return conn


def seed_demo_data(conn):
    """Creates five sample villages with 90 days of simulated history."""
    rng = np.random.default_rng(42)
    villages = [
        ("Hosahalli", 50000, 120, 1.00),
        ("Kurubarahalli", 32000, 85, 0.90),
        ("Nagenahalli", 70000, 160, 1.00),   # has a simulated leak
        ("Gowdanapalya", 25000, 60, 0.70),   # weak supply, at risk
        ("Thimmapura", 45000, 110, 0.95),
    ]
    start = dt.date.today() - dt.timedelta(days=89)
    for vid, (name, cap, hh, supply_factor) in enumerate(villages, start=1):
        conn.execute(
            "INSERT INTO villages(name, capacity_l, households) VALUES(?,?,?)",
            (name, cap, hh),
        )
        level = cap * 0.7
        for d in range(90):
            day = start + dt.timedelta(days=d)
            rain = float(max(0.0, rng.normal(4, 5))) if rng.random() < 0.3 else 0.0
            supply = hh * 90 * supply_factor * rng.uniform(0.9, 1.1) if d % 2 == 0 else 0.0
            cons = hh * PERSONS_PER_HOUSEHOLD * 9 * rng.uniform(0.9, 1.1)
            if vid == 3 and 60 <= d <= 66:   # simulated leak
                cons += hh * 30
            level = min(cap, max(0.0, level + supply + rain * cap * RAIN_RUNOFF - cons))
            conn.execute(
                "INSERT INTO daily(village_id, day, level_l, supply_l, consumption_l, rainfall_mm)"
                " VALUES(?,?,?,?,?,?)",
                (vid, day.isoformat(), level, supply, cons, rain),
            )
    conn.commit()


@st.cache_data(ttl=60)
def load_villages():
    return pd.read_sql("SELECT * FROM villages ORDER BY id", get_conn())


@st.cache_data(ttl=60)
def load_daily(village_id):
    return pd.read_sql(
        "SELECT * FROM daily WHERE village_id=? ORDER BY day",
        get_conn(), params=(int(village_id),), parse_dates=["day"],
    )


def add_alert(conn, village_id, severity, message):
    exists = conn.execute(
        "SELECT 1 FROM alerts WHERE village_id=? AND message=? AND resolved=0",
        (int(village_id), message),
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO alerts(village_id, created, severity, message) VALUES(?,?,?,?)",
            (int(village_id), dt.date.today().isoformat(), severity, message),
        )
        conn.commit()


# ---------------------------------------------------------------- analytics
def scenario(v, df, rain_m=1.0, supply_m=1.0, cons_m=1.0):
    """Projects net daily water change from 30-day averages with optional what-if multipliers."""
    recent = df.tail(30)
    level = float(df.level_l.iloc[-1])
    cons = recent.consumption_l.mean() * cons_m
    supply = recent.supply_l.mean() * supply_m
    rain_gain = recent.rainfall_mm.mean() * rain_m * v["capacity_l"] * RAIN_RUNOFF
    net = supply + rain_gain - cons
    days_left = level / -net if net < 0 else None
    return {
        "level": level, "pct": level / v["capacity_l"] * 100,
        "cons": cons, "supply": supply, "rain_gain": rain_gain,
        "net": net, "days_left": days_left,
    }


def assess(v, df):
    s = scenario(v, df)
    d = s["days_left"]
    if s["pct"] < 15 or (d is not None and d < 3):
        s["status"] = "Critical"
    elif s["pct"] < 35 or (d is not None and d < 7):
        s["status"] = "Warning"
    else:
        s["status"] = "Normal"
    return s


def observed_trend(df):
    """Learned linear trend (L/day) of the last 30 days of tank level, via scikit-learn."""
    y = df.level_l.tail(30).values
    X = np.arange(len(y)).reshape(-1, 1)
    return float(LinearRegression().fit(X, y).coef_[0])


def recommendations(v, df):
    s = assess(v, df)
    tips = []
    if s["days_left"] is not None:
        tips.append(
            f"If current consumption continues and no additional supply arrives, "
            f"{v['name']} may run short in about {s['days_left']:.0f} days."
        )
    else:
        tips.append("Supply currently covers consumption; no shortage is projected at current rates.")
    if s["status"] != "Normal":
        tips.append("Reduce non-essential use (washing, bathing) and schedule an extra tanker or borewell supply.")
    recent = df.tail(14)
    med = df.consumption_l.median()
    if (recent.consumption_l > 1.3 * med).any():
        tips.append("Consumption spikes were seen recently. Inspect pipes and taps for leakage.")
    return tips


def page_dashboard(conn, v, df, villages):
    st.header(f"Village dashboard: {v['name']}")
    s = assess(v, df)
    last = df.iloc[-1]
    c = st.columns(4)
    c[0].metric("Tank capacity (L)", f"{v['capacity_l']:,.0f}")
    c[1].metric("Current water (L)", f"{last.level_l:,.0f}")
    c[2].metric("Today's consumption (L)", f"{last.consumption_l:,.0f}")
    c[3].metric("Today's supply (L)", f"{last.supply_l:,.0f}")
    c = st.columns(3)
    c[0].metric("Households", int(v["households"]))
    c[1].metric("Availability", f"{s['pct']:.1f}%")
    c[2].metric("Status", s["status"])

    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=s["pct"],
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1565c0"},
                   "steps": [{"range": [0, 15], "color": "#ffcdd2"},
                             {"range": [15, 35], "color": "#fff3cd"}]},
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.plotly_chart(px.line(df, x="day", y="level_l", title="Tank level (L)"),
                        use_container_width=True)

    st.subheader("Water-level sensor")
    st.info("SIMULATED ESP32 sensor. These readings are generated, not measured. No hardware is required.")
    reading = last.level_l * (1 + np.random.default_rng().normal(0, 0.01))
    st.write(f"Simulated reading: **{reading:,.0f} L** · Connection status: **Simulated (no hardware)**")

    st.subheader("Summary")
    for tip in recommendations(v, df):
        st.write(tip)


def page_prediction(conn, v, df, villages):
    st.header("AI water shortage prediction")
    st.caption("Projection uses 30-day averages. Adjust the assumptions to run what-if scenarios.")
    c = st.columns(3)
    rain_adj = c[0].slider("Rainfall change (%)", -100, 200, 0)
    supply_adj = c[1].slider("Supply change (%)", -100, 100, 0)
    cons_adj = c[2].slider("Consumption change (%)", -50, 100, 0)

    s = scenario(v, df, 1 + rain_adj / 100, 1 + supply_adj / 100, 1 + cons_adj / 100)
    t = np.arange(31)
    levels = np.clip(s["level"] + t * s["net"], 0, v["capacity_l"])
    dates = pd.Timestamp.today().normalize() + pd.to_timedelta(t, unit="D")
    fig = px.line(x=dates, y=levels, labels={"x": "Date", "y": "Projected water (L)"},
                  title="30-day projection")
    fig.add_hline(y=v["capacity_l"] * 0.15, line_dash="dot", line_color="red",
                  annotation_text="Critical (15%)")
    st.plotly_chart(fig, use_container_width=True)

    c = st.columns(4)
    for col, n in zip(c[:3], [7, 15, 30]):
        col.metric(f"Day {n} water (L)", f"{levels[n]:,.0f}",
                   f"{levels[n] / v['capacity_l'] * 100:.1f}% of capacity")
    c[3].metric("Days before shortage",
                "No shortage projected" if s["days_left"] is None else f"{s['days_left']:.1f}")

    st.subheader("Why this prediction?")
    st.write(
        f"Average daily supply is {s['supply']:,.0f} L and consumption is {s['cons']:,.0f} L. "
        f"Estimated rainfall runoff adds {s['rain_gain']:,.0f} L per day. "
        f"The net change is {s['net']:+,.0f} L per day. "
        f"The learned 30-day trend from linear regression is {observed_trend(df):+,.0f} L per day."
    )
    st.caption("This is an estimate for planning and is not a guaranteed forecast. Validate it before using it for public allocation.")

    st.subheader("Villages at risk")
    st.dataframe(risk_table(villages), use_container_width=True, hide_index=True)


def risk_table(villages):
    rows = []
    for _, row in villages.iterrows():
        v = row.to_dict()
        s = assess(v, load_daily(v["id"]))
        rows.append({
            "Village": v["name"], "Status": s["status"],
            "Water left %": round(s["pct"], 1),
            "Days before shortage": None if s["days_left"] is None else round(s["days_left"], 1),
        })
    out = pd.DataFrame(rows)
    out["_d"] = out["Days before shortage"].fillna(9999)
    return out.sort_values(["_d", "Water left %"]).drop(columns="_d")


def page_anomaly(conn, v, df, villages):
    st.header("Leak and anomaly detection")
    X = df[["consumption_l"]].values
    model = IsolationForest(contamination=0.05, random_state=0).fit(X)
    flagged = df.assign(anomaly=model.predict(X) == -1)
    med = df.consumption_l.median()
    std = df.consumption_l.std()

    fig = px.line(df, x="day", y="consumption_l", title="Daily consumption (L)")
    a = flagged[flagged.anomaly]
    fig.add_scatter(x=a.day, y=a.consumption_l, mode="markers", marker_color="red",
                    name="Anomaly")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected anomalies")
    if a.empty:
        st.success("No unusual consumption detected for this village.")
    else:
        for _, r in a.tail(5).iterrows():
            z = (r.consumption_l - df.consumption_l.mean()) / std if std else 0
            st.warning(
                f"{r.day:%d %b %Y}: consumption {r.consumption_l:,.0f} L, "
                f"{(r.consumption_l / med - 1) * 100:.0f}% above the median of {med:,.0f} L "
                f"(z-score {z:.1f}). Possible leakage. Inspect pipes and taps."
            )
    st.subheader("Anomaly history")
    st.dataframe(a[["day", "consumption_l", "level_l"]].sort_values("day", ascending=False),
                 use_container_width=True, hide_index=True)


def page_distribution(conn, v, df, villages):
    st.header("Smart water distribution")
    s = assess(v, df)
    hh = int(v["households"])
    c = st.columns(3)
    available = c[0].number_input("Total water available (L)", min_value=0.0,
                                  value=float(round(s["level"])), step=500.0)
    days = c[1].slider("Days to stretch the water over", 1, 14, 3)
    prio = c[2].number_input("Priority households (elderly, children, pregnant women)",
                             min_value=0, max_value=hh, value=min(hh, hh // 5))
    reduction = st.slider("Reduce non-essential use by (%)", 0, 50, 0)

    budget = available / days
    ess_per_hh = PERSONS_PER_HOUSEHOLD * ESSENTIAL_LPCD
    need_ess = (prio * 1.25 + (hh - prio) * 1.0) * ess_per_hh
    ratio = min(1.0, budget / need_ess) if need_ess else 1.0
    spare = max(0.0, budget - need_ess * ratio)
    norm = NON_ESSENTIAL_NORM * (1 - reduction / 100)
    noness = min(norm, spare / hh) if hh else 0.0

    alloc = pd.DataFrame([
        {"Group": "Priority households", "Households": prio,
         "Essential (L/hh/day)": round(ess_per_hh * 1.25 * ratio, 1),
         "Non-essential (L/hh/day)": round(noness, 1),
         "Total (L/hh/day)": round(ess_per_hh * 1.25 * ratio + noness, 1)},
        {"Group": "General households", "Households": hh - prio,
         "Essential (L/hh/day)": round(ess_per_hh * ratio, 1),
         "Non-essential (L/hh/day)": round(noness, 1),
         "Total (L/hh/day)": round(ess_per_hh * ratio + noness, 1)},
    ])
    st.dataframe(alloc, use_container_width=True, hide_index=True)

    total_used = ((prio * 1.25 + (hh - prio)) * ess_per_hh * ratio) + hh * noness
    st.write(f"Daily budget: **{budget:,.0f} L**. Planned use: **{total_used:,.0f} L**. "
             f"Essential needs are met first, and the rest is shared as non-essential use.")
    if ratio < 1:
        st.warning("Water is too short to meet essential needs in full. Arrange extra supply or lengthen the rationing period.")
    else:
        st.success("Essential needs are covered at this rate.")


def page_rainfall(conn, v, df, villages):
    st.header("Rainfall and water resources")
    st.write(f"Estimated runoff gain per 10 mm of rain: **{10 * v['capacity_l'] * RAIN_RUNOFF:,.0f} L**")
    with st.form("rain_form"):
        day = st.date_input("Date", value=dt.date.today(), max_value=dt.date.today())
        mm = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
        if st.form_submit_button("Save rainfall"):
            n = conn.execute(
                "UPDATE daily SET rainfall_mm=? WHERE village_id=? AND day=?",
                (mm, int(v["id"]), day.isoformat()),
            ).rowcount
            conn.commit()
            st.cache_data.clear()
            if n:
                st.success("Rainfall saved.")
            else:
                st.warning("No water record exists for that date yet.")
    st.plotly_chart(px.bar(df, x="day", y="rainfall_mm", title="Daily rainfall (mm)"),
                    use_container_width=True)
    st.plotly_chart(px.line(df, x="day", y="level_l", title="Tank and reservoir level (L)"),
                    use_container_width=True)


def page_comparison(conn, v, df, villages):
    st.header("Village comparison")
    table = risk_table(villages)
    st.subheader("Ranked by shortage risk")
    st.dataframe(table, use_container_width=True, hide_index=True)
    urgent = table[table["Status"] == "Critical"]["Village"].tolist()
    if urgent:
        st.error("Urgent attention: " + ", ".join(urgent))
    st.plotly_chart(px.bar(table, x="Village", y="Water left %", color="Status",
                           title="Water left (% of capacity)"), use_container_width=True)
    rows = []
    for _, r in villages.iterrows():
        d = load_daily(r["id"])
        rows.append({"Village": r["name"], "Avg daily consumption (L)": round(d.consumption_l.tail(30).mean()),
                     "Avg daily supply (L)": round(d.supply_l.tail(30).mean())})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_alerts(conn, v, df, villages):
    st.header("Alerts and notifications")
    if st.button("Refresh alerts from latest data"):
        for _, row in villages.iterrows():
            vd = row.to_dict()
            d = load_daily(vd["id"])
            s = assess(vd, d)
            if s["status"] in ("Critical", "Warning"):
                msg = f"{s['status']}: {s['pct']:.0f}% water left"
                if s["days_left"] is not None:
                    msg += f", about {s['days_left']:.0f} days before shortage"
                add_alert(conn, vd["id"], s["status"], msg)
            med = d.consumption_l.median()
            for _, r in d.tail(14).iterrows():
                if r.consumption_l > 1.3 * med:
                    add_alert(conn, vd["id"], "Anomaly",
                              f"Unusual use on {r.day:%d %b}: {r.consumption_l:,.0f} L vs median {med:,.0f} L")
        st.cache_data.clear()
        st.rerun()

    alerts = pd.read_sql(
        "SELECT a.id, v.name AS village, a.created, a.severity, a.message, a.resolved "
        "FROM alerts a JOIN villages v ON v.id = a.village_id "
        "WHERE a.village_id=? ORDER BY a.id DESC", conn, params=(int(v["id"]),),
    )
    open_alerts = alerts[alerts.resolved == 0]
    st.subheader(f"Open alerts for {v['name']}: {len(open_alerts)}")
    if open_alerts.empty:
        st.success("No open alerts.")
    for _, r in open_alerts.iterrows():
        c = st.columns([5, 1])
        if r.severity == "Critical":
            c[0].error(f"{r.created} · {r.message}")
        elif r.severity == "Warning":
            c[0].warning(f"{r.created} · {r.message}")
        else:
            c[0].info(f"{r.created} · {r.message}")
        if c[1].button("Resolve", key=f"res{r.id}"):
            conn.execute("UPDATE alerts SET resolved=1 WHERE id=?", (int(r.id),))
            conn.commit()
            st.rerun()
    with st.expander("Alert history"):
        st.dataframe(alerts, use_container_width=True, hide_index=True)


def page_reports(conn, v, df, villages):
    st.header("Reports and analytics")
    period = st.selectbox("Period", ["Weekly", "Monthly"])
    rule = "W" if period == "Weekly" else "ME"
    g = (df.set_index("day")
           .resample(rule)
           .agg(avg_level_l=("level_l", "mean"), total_supply_l=("supply_l", "sum"),
                total_consumption_l=("consumption_l", "sum"), rain_mm=("rainfall_mm", "sum"))
           .round(0).reset_index())
    g["day"] = g["day"].dt.date
    st.dataframe(g, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(g, x="day", y=["total_supply_l", "total_consumption_l"],
                           barmode="group", title=f"{period} supply vs consumption"),
                    use_container_width=True)
    st.subheader("Recommendations")
    for tip in recommendations(v, df):
        st.write(tip)
    st.download_button("Download daily data (CSV)", df.to_csv(index=False).encode(),
                       f"{v['name']}_daily.csv", "text/csv")
    st.download_button("Download period summary (CSV)", g.to_csv(index=False).encode(),
                       f"{v['name']}_{period.lower()}.csv", "text/csv")


def page_settings(conn, v, df, villages):
    st.header("Settings and about")
    st.subheader("Village configuration")
    with st.form("cfg_form"):
        cap = st.number_input("Tank capacity (L)", min_value=1000.0,
                              value=float(v["capacity_l"]), step=500.0)
        hh = st.number_input("Households", min_value=1, value=int(v["households"]), step=1)
        if st.form_submit_button("Save"):
            conn.execute("UPDATE villages SET capacity_l=?, households=? WHERE id=?",
                         (cap, int(hh), int(v["id"])))
            conn.commit()
            st.cache_data.clear()
            st.success("Saved.")
    st.subheader("Data management")
    if st.button("Reset all data to demo sample"):
        conn.executescript("DELETE FROM daily; DELETE FROM alerts; DELETE FROM villages;")
        seed_demo_data(conn)
        st.cache_data.clear()
        st.rerun()
    st.subheader("About")
    st.write("JalRakshak AI: intelligent rural water management. Local algorithms only, "
             "no paid APIs. Predictions are estimates and need validation before public use. "
             "Sensor readings are simulated unless a real device is connected.")


PAGES = {
    "Village Dashboard": page_dashboard,
    "Shortage Prediction": page_prediction,
    "Leak & Anomaly Detection": page_anomaly,
    "Smart Distribution": page_distribution,
    "Rainfall & Resources": page_rainfall,
    "Village Comparison": page_comparison,
    "Alerts": page_alerts,
    "Reports & Analytics": page_reports,
    "Settings & About": page_settings,
}


def main():
    conn = get_conn()
    villages = load_villages()
    names = dict(zip(villages["id"], villages["name"]))
    st.sidebar.title("💧 JalRakshak AI")
    vid = st.sidebar.selectbox("Village", list(names), format_func=names.get)
    page = st.sidebar.radio("Go to", list(PAGES))
    v = villages[villages["id"] == vid].iloc[0].to_dict()
    df = load_daily(vid)
    st.markdown(
        '<div class="jal-banner"><h2>JalRakshak AI</h2>'
        '<p>Intelligent Rural Water Management · Sample data for demonstration</p></div>',
        unsafe_allow_html=True,
    )
    PAGES[page](conn, v, df, villages)


main()
