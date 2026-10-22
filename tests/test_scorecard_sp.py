"""Tests for the scorecardpy engine. Skipped automatically if scorecardpy is not installed."""
import numpy as np
import pytest

pytest.importorskip("scorecardpy")

from credit_risk.config import load_config                      # noqa: E402
from credit_risk.evaluate import band_table, is_monotone_decreasing_risk   # noqa: E402
from credit_risk.scorecard_sp import SPScorecard                 # noqa: E402
try:
    from .test_scorecard import _frame                           # noqa: E402
except ImportError:
    from test_scorecard import _frame                            # noqa: E402

CFG = load_config()


def _fit():
    X, y = _frame()
    return SPScorecard(CFG).fit(X, y), X, y


def test_sp_real_signal_kept_and_noise_dropped():
    sc, _, _ = _fit()
    assert "CreditScore" in sc.selected_
    for noise in ("Occupancy", "Channel", "PPM"):
        assert noise not in sc.selected_


def test_sp_all_coefficients_positive_under_scorecardpy_convention():
    sc, _, _ = _fit()
    assert (sc.model_.coef_[0] > 0).all()


def test_sp_riskiest_credit_score_bin_has_highest_woe():
    sc, _, _ = _fit()
    b = sc.bins_["CreditScore"]
    lowest_score_bin = b[b["bin"].astype(str).str.startswith("[-inf")]
    assert lowest_score_bin["woe"].iloc[0] == b["woe"].max()


def test_sp_score_is_exact_function_of_pd():
    sc, X, _ = _fit()
    p = sc.predict_pd(X)
    assert np.allclose(sc.score(X), sc.offset + sc.factor * np.log((1 - p) / p))


def test_sp_higher_credit_score_means_lower_pd():
    sc, X, _ = _fit()
    lo, hi = X.head(200).copy(), X.head(200).copy()
    lo["CreditScore"], hi["CreditScore"] = 550, 800
    assert sc.predict_pd(hi).mean() < sc.predict_pd(lo).mean()


def test_sp_risk_bands_are_monotone_on_train():
    sc, X, y = _fit()
    t = band_table(y, sc.band(sc.score(X)), sc.band_names)
    assert is_monotone_decreasing_risk(t)


def test_sp_robust_to_missing_values_at_scoring_time():
    sc, X, _ = _fit()
    Z = X.head(5).copy()
    Z.loc[Z.index[0], "CreditScore"] = np.nan
    Z.loc[Z.index[1], "DTI"] = np.nan
    p = sc.predict_pd(Z)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def test_sp_points_table_has_base_points_and_every_selected_feature():
    sc, _, _ = _fit()
    pt = sc.points_table()
    assert "(base points)" in set(pt.feature)
    assert set(sc.selected_) <= set(pt.feature)
    assert pt["points"].notna().all()


def test_sp_reason_codes_one_list_per_row():
    sc, X, _ = _fit()
    reasons = sc.reason_codes(X.head(10), top_n=3)
    assert len(reasons) == 10 and all(len(r) <= 3 for r in reasons)


def test_sp_save_and_load_roundtrip(tmp_path):
    sc, X, _ = _fit()
    path = tmp_path / "sp.joblib"
    sc.save(path)
    again = SPScorecard.load(path)
    assert np.allclose(sc.predict_pd(X.head(50)), again.predict_pd(X.head(50)))
