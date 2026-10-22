"""PSI, performance monitoring and alerts."""
import numpy as np
import pandas as pd

from credit_risk import monitor as M
from credit_risk.config import load_config

from .test_scorecard import _fit, _frame

CFG = load_config()


def test_psi_identical_samples_is_near_zero():
    x = np.random.default_rng(0).normal(size=20_000)
    y = np.random.default_rng(1).normal(size=20_000)
    assert M.psi(x, y) < 0.01


def test_psi_shifted_sample_is_large():
    x = np.random.default_rng(0).normal(size=20_000)
    assert M.psi(x, x + 1.5) > 0.25


def test_psi_is_zero_for_the_same_sample_and_never_negative():
    x = np.random.default_rng(0).normal(size=5_000)
    assert M.psi(x, x) == 0
    assert M.psi(x, x[::-1] * 1.1) >= 0


def test_psi_categorical_flags_new_category():
    base = pd.Series(["A"] * 700 + ["B"] * 300)
    same = pd.Series(["A"] * 690 + ["B"] * 310)
    new = pd.Series(["A"] * 400 + ["B"] * 300 + ["Z"] * 300)
    assert M.psi_categorical(base, same) < 0.01
    assert M.psi_categorical(base, new) > 0.25


def test_missing_values_get_their_own_bin():
    ref = np.r_[np.random.default_rng(0).normal(size=1000)]
    edges = M._inner_edges(ref, 10)
    sh = M._numeric_shares(np.r_[ref[:900], [np.nan] * 100], edges)
    assert abs(sh.sum() - 1) < 1e-9 and sh[-1] == 0.1


def test_stable_batch_is_stable_and_shifted_batch_is_caught():
    sc, X, y = _fit()
    base = M.build_baseline(sc, CFG, X, X, y)
    X2, y2 = _frame(seed=2)
    row, fdf, alerts = M.evaluate_batch(sc, base, X2, y2, "same", CFG)
    assert row["status"] == "stable" and row["score_psi"] < 0.05
    assert 0.85 < row["calibration_ratio"] < 1.15

    X3 = X2.copy()
    X3["CreditScore"] = X3["CreditScore"] - 60
    row3, fdf3, alerts3 = M.evaluate_batch(sc, base, X3, y2, "shifted", CFG)
    assert row3["score_psi"] > 0.25 and row3["status"] == "alert"
    assert row3["max_feature_name"] == "CreditScore"
    assert any(a["metric"] == "score_psi" for a in alerts3)


def test_lost_signal_triggers_auc_and_calibration_alerts():
    sc, X, y = _fit()
    base = M.build_baseline(sc, CFG, X, X, y)
    X2, _ = _frame(seed=3)
    rng = np.random.default_rng(5)
    y_random = (rng.random(len(X2)) < 0.15).astype(int)     # outcomes unrelated to the model, higher rate
    row, _, alerts = M.evaluate_batch(sc, base, X2, y_random, "broken", CFG)
    metrics = {a["metric"] for a in alerts}
    assert "auc_drop" in metrics and "calibration" in metrics
    assert row["status"] == "alert"


def test_small_batch_does_not_report_auc():
    sc, X, y = _fit()
    base = M.build_baseline(sc, CFG, X, X, y)
    row, _, _ = M.evaluate_batch(sc, base, X.iloc[:300], y[:300], "tiny", CFG)
    assert np.isnan(row["auc"]) and np.isnan(row["calibration_ratio"])


def test_baseline_roundtrips_through_json(tmp_path):
    sc, X, y = _fit()
    base = M.build_baseline(sc, CFG, X, X, y)
    M.save_baseline(base, tmp_path / "b.json")
    again = M.load_baseline(tmp_path / "b.json")
    row1, *_ = M.evaluate_batch(sc, base, X, y, "a", CFG)
    row2, *_ = M.evaluate_batch(sc, again, X, y, "a", CFG)
    assert abs(row1["score_psi"] - row2["score_psi"]) < 1e-12


def test_run_monitoring_writes_all_outputs(tmp_path):
    """Plumbing test on the raw-format fixture: real cleaning code, batch + outcome files, all outputs written."""
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "loans_sample.csv"
    raw = pd.read_csv(fixture, dtype=str)
    (tmp_path / "batches").mkdir()
    (tmp_path / "outcomes").mkdir()
    raw.drop(columns=["EverDelinquent", "MonthsDelinquent", "MonthsInRepayment"]).to_csv(
        tmp_path / "batches" / "loans_2000_01.csv", index=False)
    raw[["LoanSeqNum", "EverDelinquent", "MonthsDelinquent"]].to_csv(
        tmp_path / "outcomes" / "outcomes_2000_01.csv", index=False)
    cfg = {**CFG, "monitoring": {"min_batch_loans": 10, "min_bads_for_auc": 1}}
    sc, X, y = _fit()
    base = M.build_baseline(sc, cfg, X, X, y)
    summary, feat, alerts = M.run_monitoring(sc, base, cfg, tmp_path / "batches", tmp_path / "outcomes", tmp_path / "out")
    for f in ("batch_monitoring.csv", "feature_psi.csv", "alerts.json", "monitoring_trend.png"):
        assert (tmp_path / "out" / f).exists()
    assert list(summary["batch"]) == ["2000_01"] and len(feat) == len(base["features"])
