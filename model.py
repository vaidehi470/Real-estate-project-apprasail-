"""Pure financial model (no Streamlit) so it can be unit-tested.
All money in INR crore."""
import numpy as np, pandas as pd


import re

def clean_promoter(name):
    """Normalise developer names: case, punctuation, legal suffixes (Ltd/Limited/Pvt/Private...)."""
    n = re.sub(r"[.,&]", " ", str(name).upper())
    n = re.sub(r"\b(LIMITED|LTD)\b", "LTD", n)
    n = re.sub(r"\b(PRIVATE|PVT)\b", "PVT", n)
    n = re.sub(r"\bLLP\b", "LLP", n)
    return " ".join(n.split())

def irr(cf):
    """Annualised IRR from monthly cash flows. Scans for the first sign change of NPV(r), then bisects."""
    cf = np.asarray(cf, float)
    if cf.min() >= 0 or cf.max() <= 0: return np.nan
    t = np.arange(len(cf)); f = lambda r: (cf / (1 + r) ** t).sum()
    grid = np.r_[np.linspace(-0.05, 0.05, 101), np.linspace(0.06, 1.0, 95)]
    vals = [f(r) for r in grid]
    for i in range(len(grid) - 1):
        if vals[i] == 0: return (1 + grid[i]) ** 12 - 1
        if vals[i] * vals[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(60):
                m = (lo + hi) / 2
                if f(lo) * f(m) <= 0: hi = m
                else: lo = m
            return (1 + m) ** 12 - 1
    return np.nan

def shape(n, kind):
    x = np.linspace(2, .5, n) if kind == "Front-loaded" else np.linspace(.5, 2, n) if kind == "Back-loaded" else np.ones(n)
    return x / x.sum()

def model(p):
    A, L, S = int(p["approval_m"]), int(p["launch_m"]), int(p["sales_m"])
    C = int(p["constr_m"] + p["delay_m"]); T = max(A + C, L + S) + 3; n = T + 1; t = np.arange(n)
    esc = (1 + p["cost_esc"]) ** (t / 12)
    land = np.zeros(n); land[0] = p["land_acres"] * p["land_cr_acre"] * (1 + p["stamp_pct"])
    appr = np.zeros(n); appr[:max(A, 1)] = p["approval_cr"] / max(A, 1)
    bua = p["land_acres"] * 43560 * p["far"]; sal = bua * p["eff"]
    w = np.sin(np.pi * (np.arange(C) + .5) / C); w /= w.sum()                 # S-curve spend
    cons = np.zeros(n); cons[A + 1:A + 1 + C] = w * bua * p["cost_psf"] * (1 + p["contingency"]) / 1e7; cons *= esc
    prog = np.cumsum(np.r_[np.zeros(A + 1), w, np.zeros(n - A - 1 - C)])
    sw = np.zeros(n); sw[L:L + S] = shape(S, p["curve"])
    booked = sw * sal * p["price_psf"] * (1 + p["price_g"]) ** (t / 12) / 1e7
    mkt = booked * p["mkt_pct"]; ovh = np.zeros(n); ovh[:A + C + 1] = p["overhead_cr_m"]
    cost = land + appr + cons + mkt + ovh
    # collections: booking % + construction-linked % + possession % (RERA-style milestone plan)
    poss = 1 - p["book_pct"] - p["cp_pct"]; end = A + C; coll = np.zeros(n); cum = 0
    for i in range(n):
        dp = prog[i] - (prog[i - 1] if i else 0)
        fr = p["book_pct"] + p["cp_pct"] * prog[i] + (poss if i >= end else 0)
        coll[i] = booked[i] * fr + cum * (p["cp_pct"] * dp + (poss if i == end else 0)); cum += booked[i]
    rev, tc = booked.sum(), cost.sum()
    # unlevered (project) cash flow
    proj = coll - cost; tax_u = p["tax"] * max(0, rev - tc); proj[-1] -= tax_u
    # construction finance: debt funds LTC% of costs, interest capitalised, cash-sweep repayment
    bal = 0; eq = np.zeros(n); dbal = np.zeros(n); intr = 0
    for i in range(n):
        it = bal * p["rate"] / 12; bal += it; intr += it
        draw = p["ltc"] * (cost[i] if p["debt_land"] else cost[i] - land[i])
        avail = coll[i] - cost[i] + draw; rep = min(bal + draw, max(avail, 0))
        bal += draw - rep; eq[i] = avail - rep; dbal[i] = bal
    tax_l = p["tax"] * max(0, rev - tc - intr); eq[-1] -= tax_l + bal
    rm = (1 + p["disc"]) ** (1 / 12) - 1; npv = (proj / (1 + rm) ** t).sum()
    cumc = np.cumsum(proj); neg = np.where(cumc < 0)[0]; pb = np.nan
    if len(neg):
        a = np.where((cumc >= 0) & (t > neg[0]))[0]; pb = a[0] if len(a) else np.nan
    inj = -eq[eq < 0].sum()
    return dict(npv=npv, irr=irr(proj), eq_irr=irr(eq), payback=pb, rev=rev, cost=tc, interest=intr, tax=tax_l,
                profit=rev - tc - intr - tax_l, poc=(rev - tc - intr - tax_l) / (tc + intr),
                peak_debt=dbal.max(), peak_eq=-np.cumsum(eq).min(), moic=eq[eq > 0].sum() / inj if inj else np.nan,
                df=pd.DataFrame(dict(month=t, land=land, approvals=appr, construction=cons, marketing=mkt,
                                     overheads=ovh, collections=coll, project_cf=proj, equity_cf=eq, debt_balance=dbal)),
                bua=bua, sal=sal)

def breakeven(p):
    f = lambda x: model({**p, "price_psf": x})["npv"]; lo, hi = 500., 80000.
    if f(hi) < 0: return np.nan
    if f(lo) > 0: return lo
    for _ in range(40):
        m = (lo + hi) / 2
        if f(m) < 0: lo = m
        else: hi = m
    return m

def shock(p, price=1, cost=1, sales=1, delay=0, rate=0):
    return {**p, "price_psf": p["price_psf"] * price, "cost_psf": p["cost_psf"] * cost,
            "sales_m": max(6, round(p["sales_m"] * sales)), "delay_m": p["delay_m"] + delay,
            "rate": max(0, p["rate"] + rate)}

def monte_carlo(p, n=400, seed=7):
    r = np.random.default_rng(seed); out = []
    for _ in range(n):
        q = shock(p, r.triangular(.85, 1, 1.1), r.triangular(.95, 1, 1.2), r.triangular(.8, 1, 1.6),
                  int(r.triangular(0, 0, 12)), r.triangular(-.01, 0, .02))
        out.append(model(q)["npv"])
    return np.array(out)

