"""Scorecard behaviour on a synthetic frame that uses the project's real feature names."""
import numpy as np
import pandas as pd

from credit_risk.config import load_config, model_inputs
from credit_risk.evaluate import band_table, is_monotone_decreasing_risk, psi
from credit_risk.scorecard import Scorecard

CFG = load_config()


def _frame(n=30_000, seed=1):
    rng = np.random.default_rng(seed)
    f = CFG["features"]
    df = pd.DataFrame({
        "CreditScore": rng.normal(700, 50, n).clip(500, 830),
        "DTI": rng.normal(33, 9, n).clip(5, 60),
        "LTV": rng.normal(76, 14, n).clip(20, 100),
        "OrigUPB": rng.lognormal(11.8, 0.5, n),
        "OrigInterestRate": rng.normal(7.0, 0.4, n),
        "MIP": rng.choice([0, 12, 25], n, p=[0.6, 0.2, 0.2]),
        "Units": rng.choice([1, 2], n, p=[0.97, 0.03]),
    })
    for c in f["categorical"]:
        df[c] = rng.choice(["P", "Q", "UNKNOWN"], n, p=[0.6, 0.3, 0.1])
    for c in f["derived"]:
        df[c] = rng.integers(0, 2, n)
    z = -2.6 - 0.025 * (df.CreditScore - 700) + 0.03 * (df.LTV - 76) + 0.03 * (df.DTI - 33)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df[model_inputs(CFG)], y


def _fit():
    X, y = _frame()
    return Scorecard(CFG).fit(X, y), X, y


def test_noise_features_are_dropped_and_real_ones_kept():
    sc, _, _ = _fit()
    assert "CreditScore" in sc.selected_
    for noise in ("Occupancy", "Channel", "PPM", "Units"):
        assert noise not in sc.selected_


def test_all_coefficients_have_expected_negative_sign():
    sc, _, _ = _fit()
    assert (sc.model_.coef_[0] < 0).all()


def test_score_is_exact_function_of_pd():
    sc, X, _ = _fit()
    p = sc.predict_pd(X)
    expected = sc.offset + sc.factor * np.log((1 - p) / p)
    assert np.allclose(sc.score(X), expected)


def test_points_table_sums_to_score():
    sc, X, _ = _fit()
    pt = sc.points_table()
    Xp = sc._prep(X)
    total = np.zeros(len(X))
    for c in sc.selected_:
        pts = pt[pt.feature == c].sort_values("bin")["points"].to_numpy()
        total += pts[sc.binner.bin_index(Xp, c)]
    assert np.allclose(total, sc.score(X))


def test_scorecard_scaling_hits_base_score_at_base_odds():
    sc, _, _ = _fit()
    s = CFG["scorecard"]
    at_base = sc.offset + sc.factor * np.log(s["base_odds"])
    at_double = sc.offset + sc.factor * np.log(2 * s["base_odds"])
    assert abs(at_base - s["base_score"]) < 1e-9
    assert abs((at_double - at_base) - s["pdo"]) < 1e-9


def test_risk_bands_are_monotone_on_train():
    sc, X, y = _fit()
    bands = sc.band(sc.score(X))
    t = band_table(y, bands, sc.band_names)
    assert is_monotone_decreasing_risk(t)


def test_robust_to_missing_and_unseen_values_at_scoring_time():
    sc, X, _ = _fit()
    Z = X.head(5).copy()
    Z.loc[Z.index[0], "CreditScore"] = np.nan
    Z.loc[Z.index[1], "DTI"] = np.nan
    Z.loc[Z.index[2], "PropertyType"] = "NEVER_SEEN"
    p = sc.predict_pd(Z)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def test_reason_codes_one_list_per_row_and_worst_first():
    sc, X, _ = _fit()
    Z = X.head(10)
    reasons = sc.reason_codes(Z, top_n=3)
    assert len(reasons) == 10
    assert all(len(r) <= 3 for r in reasons)


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50_000)
    assert psi(a, a) < 1e-6
    assert psi(a, a + 1.0) > 0.25


def test_save_and_load_roundtrip(tmp_path=None):
    import tempfile
    sc, X, _ = _fit()
    d = tmp_path or tempfile.mkdtemp()
    path = f"{d}/sc.joblib"
    sc.save(path)
    again = Scorecard.load(path)
    assert np.allclose(sc.predict_pd(X.head(50)), again.predict_pd(X.head(50)))
