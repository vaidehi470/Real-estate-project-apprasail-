"""Real Estate Project Feasibility & Financial Appraisal Tool
Run:  streamlit run app.py
All money in INR crore (1 cr = 10 million). Monthly cash-flow model."""
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from model import model, breakeven as _be, monte_carlo as _mc, shock, irr, clean_promoter


# Streamlit < 1.50 uses use_container_width=True; newer versions use width="stretch"
_v = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
W = dict(width="stretch") if _v >= (1, 50) else dict(use_container_width=True)

st.set_page_config(page_title="RE Feasibility Tool", layout="wide", page_icon="🏗️")
DATA = Path(__file__).parent / "karnataka_rera_hoskote_whitefield_budigere.xlsx"

# ───────────────────────── RERA market data ─────────────────────────
@st.cache_data
def load(src):
    d = pd.read_excel(src, sheet_name="All_Localities")
    d["approved"] = pd.to_datetime(d.approved_on, dayfirst=True, errors="coerce")
    d["completion"] = pd.to_datetime(d.proposed_completion_date, dayfirst=True, errors="coerce")
    d["year"] = d.approved.dt.year
    d["status_c"] = d.status.str.upper().str.extract(r"^(APPROVED|TRANSFERED|REJECTED|WITHDRAWN)")[0]
    d["dur_yrs"] = (d.completion - d.approved).dt.days / 365.25
    d.loc[(d.dur_yrs < 0.5) | (d.dur_yrs > 10), "dur_yrs"] = np.nan
    d["promoter"] = d.promoter_name.map(clean_promoter)
    return d

src = DATA if DATA.exists() else st.sidebar.file_uploader("Upload RERA xlsx", type="xlsx")
if src is None:
    st.warning("Place the RERA .xlsx next to app.py or upload it in the sidebar."); st.stop()
rera = load(src)


breakeven = st.cache_data(_be)
monte_carlo = st.cache_data(_mc)

# ───────────────────────── Sidebar inputs ─────────────────────────
st.sidebar.title("Project assumptions")
st.sidebar.caption("Defaults are ILLUSTRATIVE – replace with your own land/cost/price data. ₹ crore unless stated.")
loc = st.sidebar.selectbox("Micro-market (benchmarks from RERA data)", ["All", *sorted(rera.locality.unique())])
sub = rera if loc == "All" else rera[rera.locality == loc]
res_med = sub[sub.project_type.str.contains("Residential", na=False)].dur_yrs.median()
default_c = int(np.clip(round((res_med if pd.notna(res_med) else 3.5) * 12), 24, 60))
P = {}
with st.sidebar.expander("Land & product", True):
    P["land_acres"] = st.number_input("Land (acres)", 0.5, 100., 4.0, 0.5)
    P["land_cr_acre"] = st.number_input("Land price (₹ cr/acre)", 0.5, 100., 15.0, 0.5)
    P["stamp_pct"] = st.slider("Stamp duty + registration + brokerage %", 0., 12., 7.0) / 100
    P["far"] = st.slider("FAR / FSI", 1.0, 4.0, 2.0, 0.05)
    P["eff"] = st.slider("Saleable / built-up efficiency", 0.5, 1.0, 0.75, 0.01)
with st.sidebar.expander("Costs & approvals", True):
    P["cost_psf"] = st.number_input("Construction cost (₹/sq ft BUA)", 1000, 10000, 2800, 100)
    P["contingency"] = st.slider("Contingency %", 0., 15., 5.) / 100
    P["cost_esc"] = st.slider("Cost escalation % p.a.", 0., 12., 5.) / 100
    P["approval_cr"] = st.number_input("Approvals & statutory fees (₹ cr)", 0., 100., 6.0, 0.5)
    P["mkt_pct"] = st.slider("Marketing + brokerage % of sales", 0., 10., 3.) / 100
    P["overhead_cr_m"] = st.number_input("Overheads (₹ cr / month)", 0., 5., 0.25, 0.05)
with st.sidebar.expander("Sales & collections", True):
    P["price_psf"] = st.number_input("Sale price (₹/sq ft saleable)", 2000, 40000, 8000, 250)
    P["price_g"] = st.slider("Price growth % p.a.", 0., 12., 3.) / 100
    P["sales_m"] = st.slider("Sales period (months)", 6, 84, 36)
    P["launch_m"] = st.slider("Launch month (from land purchase)", 0, 24, 6)
    P["curve"] = st.selectbox("Sales phasing", ["Linear", "Front-loaded", "Back-loaded"])
    P["book_pct"] = st.slider("Booking amount %", 5., 30., 10.) / 100
    P["cp_pct"] = st.slider("Construction-linked %", 40., 90., 80., 5.) / 100
with st.sidebar.expander("Timeline", True):
    P["approval_m"] = st.slider("Approval period (months)", 1, 24, 9)
    P["constr_m"] = st.slider(f"Construction (months) – RERA median for {loc}: {default_c}", 12, 84, default_c,
                              key=f"c_{loc}")
    P["delay_m"] = st.slider("Delay stress (months)", 0, 24, 0)
with st.sidebar.expander("Finance & tax", True):
    P["ltc"] = st.slider("Debt funding % of cost", 0., 80., 40.) / 100
    P["debt_land"] = st.checkbox("Debt also funds land", False)
    P["rate"] = st.slider("Interest rate % p.a.", 6., 20., 12.) / 100
    P["disc"] = st.slider("Discount rate (WACC/hurdle) % p.a.", 8., 30., 16.) / 100
    P["tax"] = st.slider("Income tax on profit %", 0., 40., 25.17) / 100
    min_poc = st.slider("Min. profit-on-cost hurdle %", 5., 40., 20.) / 100

# ───────────────────────── Compute ─────────────────────────
R = model(P); BE = breakeven(P); mc = monte_carlo(P)
p_loss = (mc < 0).mean(); df = R["df"]; cr = lambda x: f"₹{x:,.1f} cr"; pct = lambda x: "n/a" if pd.isna(x) else f"{x:.1%}"
checks = {
    f"Project IRR ≥ {P['disc']:.0%} hurdle": (R["irr"] >= P["disc"]) if pd.notna(R["irr"]) else False,
    "NPV > 0": R["npv"] > 0,
    f"Profit on cost ≥ {min_poc:.0%}": R["poc"] >= min_poc,
    "Break-even price ≥10% below plan": (pd.notna(BE) and BE <= 0.9 * P["price_psf"]),
    "P(NPV<0) under risk simulation < 25%": p_loss < 0.25}
score = sum(checks.values())
verdict = ("GO", "success") if score == 5 else ("CONDITIONAL", "warning") if score >= 3 else ("NO-GO", "error")

st.title("Real Estate Project Feasibility & Financial Appraisal")
st.caption(f"Micro-market benchmark: {loc} · Saleable area {R['sal']:,.0f} sq ft · Built-up {R['bua']:,.0f} sq ft · Horizon {len(df)-1} months")
t1, t2, t3, t4, t5 = st.tabs(["Go / No-Go", "Cash flow & funding", "Sensitivity & scenarios", "Downside risk", "Market context (RERA)"])

with t1:
    getattr(st, verdict[1])(f"### Recommendation: {verdict[0]}  ({score}/5 tests passed)")
    k = st.columns(5)
    k[0].metric("NPV", cr(R["npv"])); k[1].metric("Project IRR", pct(R["irr"])); k[2].metric("Equity IRR", pct(R["eq_irr"]))
    k[3].metric("Profit on cost", pct(R["poc"])); k[4].metric("Payback", "n/a" if pd.isna(R["payback"]) else f"{int(R['payback'])} months")
    k = st.columns(5)
    k[0].metric("Break-even price", "n/a" if pd.isna(BE) else f"₹{BE:,.0f}/sq ft", f"{(BE/P['price_psf']-1):.0%} vs plan" if pd.notna(BE) else None, delta_color="inverse")
    k[1].metric("Revenue", cr(R["rev"])); k[2].metric("Net profit (post-tax)", cr(R["profit"]))
    k[3].metric("Peak equity needed", cr(R["peak_eq"])); k[4].metric("Equity multiple", f"{R['moic']:.2f}x")
    c1, c2 = st.columns(2)
    c1.subheader("Test results")
    for name, ok in checks.items(): c1.write(("YES " if ok else "NO ") + name)
    comp = pd.DataFrame({"item": ["Land+duty", "Approvals", "Construction", "Marketing", "Overheads", "Interest", "Tax", "Profit"],
                         "cr": [df.land.sum(), df.approvals.sum(), df.construction.sum(), df.marketing.sum(), df.overheads.sum(),
                                R["interest"], R["tax"], max(R["profit"], 0)]})
    c2.subheader("Where each ₹ of revenue goes")
    c2.plotly_chart(px.pie(comp, names="item", values="cr", hole=.45), **W)

with t2:
    f = go.Figure()
    f.add_bar(x=df.month, y=df.collections, name="Collections", marker_color="#2a9d8f")
    f.add_bar(x=df.month, y=-(df.land + df.approvals + df.construction + df.marketing + df.overheads), name="Costs", marker_color="#e76f51")
    f.add_scatter(x=df.month, y=df.project_cf.cumsum(), name="Cumulative project CF", line=dict(color="#264653", width=3))
    f.add_scatter(x=df.month, y=df.debt_balance, name="Debt balance", line=dict(color="#e9c46a", dash="dash"))
    f.update_layout(barmode="relative", xaxis_title="Month", yaxis_title="₹ crore", height=480, legend=dict(orientation="h"))
    st.plotly_chart(f, **W)
    c = st.columns(3); c[0].metric("Peak debt", cr(R["peak_debt"])); c[1].metric("Total interest", cr(R["interest"])); c[2].metric("Interest % of cost", pct(R["interest"] / R["cost"]))
    st.dataframe(df.round(2), **W, height=250)
    st.download_button("⬇ Download cash flow (CSV)", df.to_csv(index=False), "cashflow.csv")

with t3:
    st.subheader("Tornado: NPV impact of each assumption")
    tests = [("Sale price ±20%", shock(P, price=.8), shock(P, price=1.2)), ("Construction cost ±20%", shock(P, cost=1.2), shock(P, cost=.8)),
             ("Land price ±20%", {**P, "land_cr_acre": P["land_cr_acre"] * 1.2}, {**P, "land_cr_acre": P["land_cr_acre"] * .8}),
             ("Sales pace (period ×1.5 / ×0.75)", shock(P, sales=1.5), shock(P, sales=.75)),
             ("Interest rate ±3 pts", shock(P, rate=.03), shock(P, rate=-.03)), ("Delay +12 months / none", shock(P, delay=12), P),
             ("Price growth ∓3 pts", {**P, "price_g": max(0, P["price_g"] - .03)}, {**P, "price_g": P["price_g"] + .03})]
    rows = [(n, model(a)["npv"] - R["npv"], model(b)["npv"] - R["npv"]) for n, a, b in tests]
    rows.sort(key=lambda x: max(abs(x[1]), abs(x[2])))
    tf = go.Figure()
    tf.add_bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h", name="Adverse", marker_color="#e76f51")
    tf.add_bar(y=[r[0] for r in rows], x=[r[2] for r in rows], orientation="h", name="Favourable", marker_color="#2a9d8f")
    tf.update_layout(barmode="overlay", xaxis_title="Change in NPV (₹ cr)", height=420)
    st.plotly_chart(tf, **W)
    st.info(f"Most sensitive assumption: **{rows[-1][0]}**")
    st.subheader("Scenario analysis")
    sc = {"Pessimistic": shock(P, .9, 1.1, 1.4, 9, .02), "Base": P, "Optimistic": shock(P, 1.1, .95, .8, 0, -.01)}
    sm = pd.DataFrame({k: (lambda m: {"NPV (₹ cr)": m["npv"], "Project IRR": m["irr"], "Profit on cost": m["poc"], "Peak equity (₹ cr)": m["peak_eq"]})(model(v)) for k, v in sc.items()}).T
    st.dataframe(sm.style.format({"NPV (₹ cr)": "{:,.1f}", "Project IRR": "{:.1%}", "Profit on cost": "{:.1%}", "Peak equity (₹ cr)": "{:,.1f}"}), **W)
    st.caption("Pessimistic: price −10%, cost +10%, sales 40% slower, +9 months delay, rate +2 pts. Optimistic: price +10%, cost −5%, sales 20% faster, rate −1 pt.")
    st.subheader("NPV heat-map: sale price × construction cost")
    ch = [-.15, -.1, -.05, 0, .05, .1, .15]
    z = [[model(shock(P, 1 + a, 1 + b))["npv"] for a in ch] for b in ch]
    hm = px.imshow(z, x=[f"{a:+.0%}" for a in ch], y=[f"{b:+.0%}" for b in ch], color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                   labels=dict(x="Price change", y="Cost change", color="NPV ₹cr"), text_auto=".0f", aspect="auto")
    st.plotly_chart(hm, **W)

with t4:
    st.subheader("Monte-Carlo: 400 random futures")
    st.caption("Price ×0.85–1.10, cost ×0.95–1.20, sales period ×0.8–1.6, delay 0–12 months, rate −1/+2 pts (triangular distributions).")
    c = st.columns(4); c[0].metric("P(NPV < 0)", f"{p_loss:.0%}"); c[1].metric("Median NPV", cr(np.median(mc)))
    c[2].metric("5th pct NPV (downside)", cr(np.percentile(mc, 5))); c[3].metric("95th pct NPV", cr(np.percentile(mc, 95)))
    h = px.histogram(x=mc, nbins=40, labels={"x": "NPV (₹ cr)"}, color_discrete_sequence=["#264653"]); h.add_vline(x=0, line_color="red", line_dash="dash")
    st.plotly_chart(h, **W)

with t5:
    st.subheader(f"Karnataka RERA registry – {loc}")
    st.caption("Source: uploaded RERA extract (registry data: no prices/costs). Used here for supply, timelines and competitor context.")
    k = st.columns(4); k[0].metric("Registered projects", len(sub)); k[1].metric("Median planned duration", f"{sub.dur_yrs.median():.1f} yrs")
    k[2].metric("Rejected/withdrawn", pct(sub.status_c.isin(["REJECTED", "WITHDRAWN"]).mean()))
    top5 = sub.promoter.value_counts().head(5).sum() / len(sub); k[3].metric("Top-5 developer share", pct(top5))
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(sub, x="locality", color="project_type", barmode="group", title="Supply by locality & product type"), **W)
    y = sub.groupby(["year", "locality"]).size().reset_index(name="n")
    b.plotly_chart(px.line(y, x="year", y="n", color="locality", markers=True, title="New registrations per year"), **W)
    a.plotly_chart(px.box(sub.dropna(subset=["dur_yrs"]), x="project_type", y="dur_yrs", color="locality", title="Planned approval→completion (years)"), **W)
    tp = sub.promoter.value_counts().head(10).reset_index(); tp.columns = ["promoter", "projects"]
    b.plotly_chart(px.bar(tp, x="projects", y="promoter", orientation="h", title="Most active developers (competition)").update_yaxes(autorange="reversed"), **W)
    st.dataframe(sub[["locality", "project_name", "promoter_name", "project_type", "status_c", "approved_on", "proposed_completion_date"]], **W, height=260)
