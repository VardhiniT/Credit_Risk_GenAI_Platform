import numpy as np
import pandas as pd

from credit_risk.woe import MISSING, WOEBinner


def _frame(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    cat = rng.choice(["A", "B", "C", "RARE"], n, p=[0.5, 0.3, 0.195, 0.005])
    p = 1 / (1 + np.exp(-(-2.5 + 0.9 * x + np.where(cat == "C", 0.8, 0))))   # bad rises with x
    y = (rng.random(n) < p).astype(int)
    x_m = x.copy()
    x_m[rng.random(n) < 0.03] = np.nan
    return pd.DataFrame({"x": x_m, "noise": noise, "cat": cat}), y


def _fit(**kw):
    df, y = _frame()
    b = WOEBinner(max_bins=6, min_bin_pct=0.05, **kw).fit(df, y, ["x", "noise"], ["cat"])
    return b, df, y


def test_numeric_woe_is_monotone_and_negative_trend():
    b, _, _ = _fit(monotonic=True)
    t = b.tables_["x"]
    woe = t.loc[t.label != MISSING, "woe"].to_numpy()
    assert np.all(np.diff(woe) <= 1e-12)             # bad rate rises with x -> WOE falls
    assert woe[0] > 0 > woe[-1]


def test_bins_respect_minimum_size_and_cap():
    b, _, _ = _fit()
    for c in ("x", "noise", "cat"):
        t = b.tables_[c]
        t = t[t.label != MISSING]
        assert len(t) <= 6
        assert (t["pct"] >= 0.05 - 1e-9).all() or c == "cat"   # rare categories get grouped separately


def test_informative_feature_beats_noise():
    b, _, _ = _fit()
    assert b.iv_["x"] > 0.1
    assert b.iv_["noise"] < 0.02


def test_missing_values_get_their_own_bin_and_finite_woe():
    b, df, _ = _fit()
    assert MISSING in set(b.tables_["x"]["label"])
    W = b.transform(df)
    assert np.isfinite(W.to_numpy()).all()


def test_rare_category_is_grouped_into_other():
    b, _, _ = _fit()
    labels = ",".join(b.tables_["cat"]["label"])
    assert "OTHER" in labels and "RARE" in labels


def test_unseen_category_and_unseen_missing_do_not_crash():
    b, df, _ = _fit()
    new = pd.DataFrame({"x": [np.nan, 0.0], "noise": [np.nan, 1.0], "cat": ["NEVER_SEEN", None]})
    W = b.transform(new)
    assert np.isfinite(W.to_numpy()).all()


def test_woe_sums_consistent_with_iv():
    b, _, _ = _fit()
    t = b.tables_["x"]
    assert abs(t["iv_contrib"].sum() - b.iv_["x"]) < 1e-9
