"""Run:  pytest -v test_model.py"""
import numpy as np, pytest
from model import model, breakeven, monte_carlo, shock, irr, clean_promoter

BASE = dict(land_acres=4, land_cr_acre=15, stamp_pct=.07, far=2.0, eff=.75, cost_psf=2800, contingency=.05,
            cost_esc=.05, approval_cr=6, mkt_pct=.03, overhead_cr_m=.25, price_psf=8000, price_g=.03,
            sales_m=36, launch_m=6, curve="Linear", book_pct=.10, cp_pct=.80, approval_m=9, constr_m=42,
            delay_m=0, ltc=.40, debt_land=False, rate=.12, disc=.16, tax=.2517)
P = lambda **k: {**BASE, **k}


def test_collections_equal_booked_revenue():
    for curve in ["Linear", "Front-loaded", "Back-loaded"]:
        for launch in [0, 6, 20]:
            r = model(P(curve=curve, launch_m=launch))
            assert r["df"].collections.sum() == pytest.approx(r["rev"], rel=1e-9)


def test_construction_spend_matches_inputs_without_escalation():
    r = model(P(cost_esc=0))
    expected = r["bua"] * 2800 * 1.05 / 1e7
    assert r["df"].construction.sum() == pytest.approx(expected, rel=1e-9)


def test_land_cost_includes_stamp_duty():
    assert model(BASE)["df"].land.sum() == pytest.approx(4 * 15 * 1.07)


def test_npv_monotonic_directions():
    n = lambda **k: model(P(**k))["npv"]
    assert n(price_psf=9000) > n(price_psf=8000) > n(price_psf=7000)
    assert n(cost_psf=2500) > n(cost_psf=3000)
    assert n(land_cr_acre=12) > n(land_cr_acre=18)
    assert n(delay_m=12) < n(delay_m=0)
    assert n(sales_m=24) > n(sales_m=60)


def test_irr_solver_zeroes_npv():
    cf = np.array([-100, -20, 30, 60, 80, 40.])
    r = irr(cf); rm = (1 + r) ** (1 / 12) - 1
    assert (cf / (1 + rm) ** np.arange(len(cf))).sum() == pytest.approx(0, abs=1e-6)


def test_irr_undefined_without_sign_change():
    assert np.isnan(irr([10, 20, 30])) and np.isnan(irr([-10, -5]))


def test_breakeven_price_gives_zero_npv():
    be = breakeven(BASE)
    assert model(P(price_psf=be))["npv"] == pytest.approx(0, abs=0.05)
    assert model(P(price_psf=be * 1.05))["npv"] > 0 > model(P(price_psf=be * .95))["npv"]


def test_no_debt_no_tax_equity_equals_project_cashflow():
    r = model(P(ltc=0, tax=0))
    assert np.allclose(r["df"].equity_cf, r["df"].project_cf, atol=1e-9)
    assert r["interest"] == 0


def test_debt_never_negative_and_cleared_when_profitable():
    r = model(BASE)
    assert (r["df"].debt_balance >= -1e-9).all()
    assert r["df"].debt_balance.iloc[-1] == pytest.approx(0, abs=1e-6)


def test_more_debt_means_more_interest_and_higher_peak_debt():
    lo, hi = model(P(ltc=.2)), model(P(ltc=.6))
    assert hi["interest"] > lo["interest"] and hi["peak_debt"] > lo["peak_debt"]
    assert hi["peak_eq"] < lo["peak_eq"]


def test_profit_identity():
    r = model(BASE)
    assert r["profit"] == pytest.approx(r["rev"] - r["cost"] - r["interest"] - r["tax"])


def test_tax_zero_when_loss():
    r = model(P(price_psf=3000))
    assert r["profit"] < 0 and r["tax"] == 0


@pytest.mark.parametrize("kw", [dict(ltc=0), dict(ltc=.8, debt_land=True), dict(tax=0), dict(delay_m=24),
                                dict(price_psf=2000), dict(sales_m=6, launch_m=0), dict(constr_m=12, approval_m=1),
                                dict(sales_m=84, launch_m=24, constr_m=84)])
def test_extreme_inputs_do_not_crash_or_produce_nan(kw):
    r = model(P(**kw)); d = r["df"]
    assert np.isfinite(d.drop(columns=[]).values).all()
    assert np.isfinite(r["npv"]) and np.isfinite(r["profit"])


def test_monte_carlo_reproducible_and_bracketed_by_scenarios():
    a, b = monte_carlo(BASE, n=100, seed=1), monte_carlo(BASE, n=100, seed=1)
    assert np.array_equal(a, b)
    worst = model(shock(BASE, .85, 1.2, 1.6, 12, .02))["npv"]
    best = model(shock(BASE, 1.1, .95, .8, 0, -.01))["npv"]
    assert worst - 1e-6 <= a.min() and a.max() <= best + 1e-6


def test_promoter_name_cleaning():
    assert clean_promoter("Brigade Enterprises Ltd.") == clean_promoter("BRIGADE ENTERPRISES LIMITED")
    assert clean_promoter("Mysore Projects Private Limited") == clean_promoter("Mysore Projects Pvt. Ltd")


def test_model_irr_defined_and_consistent_with_npv():
    """Regression: IRR used to return NaN when the final-month tax outflow dominated the NPV curve."""
    for kw in [{}, dict(price_psf=9500), dict(price_psf=11000), dict(cost_psf=2400), dict(tax=0)]:
        p = P(**kw); r = model(p)
        assert np.isfinite(r["irr"]), kw
        # NPV at the model's discount rate has the same sign as (IRR - discount rate)
        assert (r["npv"] > 0) == (r["irr"] > p["disc"]), kw
        # NPV evaluated at the IRR itself is ~0
        rm = (1 + r["irr"]) ** (1 / 12) - 1; cf = r["df"].project_cf.values
        assert (cf / (1 + rm) ** np.arange(len(cf))).sum() == pytest.approx(0, abs=1e-4)
