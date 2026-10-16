"""Model quality metrics: ranking (AUC, Gini, KS), calibration, gains, risk bands, PSI.

Accuracy is deliberately not reported: with a 9% bad rate, predicting "no default" for
everyone already scores 91%.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ranking_metrics(y, pd_hat):
    y = np.asarray(y)
    auc = roc_auc_score(y, pd_hat)
    fpr, tpr, _ = roc_curve(y, pd_hat)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": float(np.max(tpr - fpr)),
            "brier": float(brier_score_loss(y, pd_hat)), "bad_rate": float(y.mean()),
            "mean_pd": float(np.mean(pd_hat)), "n": int(len(y))}


def decile_table(y, pd_hat, n=10):
    """Loans sorted from riskiest to safest. Compare mean_pd with actual_rate = calibration."""
    d = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_hat)})
    d["decile"] = pd.qcut(d["pd"].rank(method="first", ascending=False), n, labels=range(1, n + 1))
    t = d.groupby("decile", observed=True).agg(loans=("y", "size"), bads=("y", "sum"),
                                               mean_pd=("pd", "mean"), actual_rate=("y", "mean")).reset_index()
    t["lift"] = t["actual_rate"] / d["y"].mean()
    t["cum_capture"] = t["bads"].cumsum() / t["bads"].sum()
    return t


def band_table(y, bands, order):
    d = pd.DataFrame({"y": np.asarray(y), "band": np.asarray(bands)})
    t = d.groupby("band").agg(loans=("y", "size"), bads=("y", "sum"), bad_rate=("y", "mean"))
    t = t.reindex(order).reset_index()
    t["share"] = t["loans"] / t["loans"].sum()
    return t


def is_monotone_decreasing_risk(table, col="bad_rate"):
    """Bands are ordered riskiest -> safest, so bad rate must fall."""
    v = table[col].to_numpy()
    return bool(np.all(np.diff(v) <= 0))


def psi(expected, actual, bins=10):
    """Population Stability Index. Rule of thumb: <0.10 stable, 0.10-0.25 watch, >0.25 shift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
