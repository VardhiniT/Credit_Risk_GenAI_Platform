"""Model monitoring: population stability (PSI), performance on matured outcomes, calibration, alerts.

Three questions are answered for every new batch of loans:

  1. Did the INPUTS change?        PSI of the score and of every feature versus the baseline.
  2. Did the model STOP WORKING?   AUC / KS on loans whose outcome is known, versus the baseline.
  3. Is the model still HONEST?    Calibration: actual bad rate divided by the average predicted PD.

PSI (Population Stability Index) compares two distributions split into the same bins:
    PSI = sum( (actual_share - expected_share) * ln(actual_share / expected_share) )
Rule of thumb: < 0.10 stable, 0.10 to 0.25 watch, > 0.25 significant shift.

The baseline (bin edges and shares from the development data) is saved once as baseline.json,
so monitoring never needs the raw training file again.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import add_targets, load_raw, prepare
from .evaluate import ranking_metrics

DEFAULTS = {"psi_watch": 0.10, "psi_alert": 0.25, "auc_drop_watch": 0.02, "auc_drop_alert": 0.05,
            "calib_watch": 0.15, "calib_alert": 0.30, "min_batch_loans": 1000, "min_bads_for_auc": 100}
EPS = 1e-4
SEVERITY = {"stable": 0, "watch": 1, "alert": 2}


def thresholds(cfg=None):
    th = dict(DEFAULTS)
    th.update((cfg or {}).get("monitoring", {}) or {})
    return th


# ----------------------------------------------------------------------------- PSI building blocks
def psi_from_shares(expected, actual):
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value, th):
    return "alert" if value >= th["psi_alert"] else "watch" if value >= th["psi_watch"] else "stable"


def _inner_edges(x, bins):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return sorted({float(v) for v in np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])})


def _numeric_shares(x, edges):
    """Share of values in each bin, plus a last bin for missing values."""
    x = np.asarray(x, float)
    miss = np.isnan(x)
    idx = np.searchsorted(edges, x[~miss], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return np.append(counts, miss.sum()) / max(len(x), 1)


def _cat_shares(s, cats):
    """Share per known category, plus a last bin for categories never seen in the baseline."""
    s = pd.Series(s).astype(str)
    vc = s.value_counts()
    known = [float(vc.get(c, 0)) for c in cats]
    other = float(vc[~vc.index.isin(cats)].sum())
    return np.array(known + [other]) / max(len(s), 1)


def psi(expected, actual, bins=10):
    """Convenience PSI between two numeric samples (bins fixed on the expected sample)."""
    edges = _inner_edges(expected, bins)
    return psi_from_shares(_numeric_shares(expected, edges), _numeric_shares(actual, edges))


def psi_categorical(expected, actual):
    cats = sorted(pd.Series(expected).astype(str).unique())
    return psi_from_shares(_cat_shares(expected, cats), _cat_shares(actual, cats))


# ----------------------------------------------------------------------------- baseline
def build_baseline(sc, cfg, X_ref, X_perf=None, y_perf=None, bins=10):
    """Freeze the development distributions. X_ref = data the model was built on (for PSI).
    X_perf / y_perf = held-out data with outcomes (for the baseline AUC and KS)."""
    f = cfg["features"]
    scores = sc.score(X_ref)
    edges = _inner_edges(scores, bins)
    base = {"created": str(dt.date.today()), "n_reference": int(len(X_ref)), "bins": bins,
            "selected_features": list(sc.selected_),
            "score": {"edges": edges, "shares": _numeric_shares(scores, edges).tolist(),
                      "mean": float(np.mean(scores))},
            "features": {}}
    bands = pd.Series(sc.band(scores)).value_counts(normalize=True)
    base["band_shares"] = {b: float(bands.get(b, 0.0)) for b in sc.band_names}
    for c in f["numeric"]:
        x = pd.to_numeric(X_ref[c], errors="coerce").to_numpy()
        e = _inner_edges(x, bins)
        base["features"][c] = {"type": "numeric", "edges": e, "shares": _numeric_shares(x, e).tolist(),
                               "median": float(np.nanmedian(x))}
    for c in list(f["categorical"]) + list(f["derived"]):
        cats = sorted(pd.Series(X_ref[c]).astype(str).unique())
        base["features"][c] = {"type": "categorical", "categories": cats,
                               "shares": _cat_shares(X_ref[c], cats).tolist()}
    if X_perf is not None and y_perf is not None:
        base["performance"] = ranking_metrics(y_perf, sc.predict_pd(X_perf))
        pb = pd.DataFrame({"band": sc.band(sc.score(X_perf)), "y": np.asarray(y_perf)}).groupby("band")["y"].mean()
        base["band_bad_rates"] = {b: float(pb.get(b, float("nan"))) for b in sc.band_names}
    return base


def save_baseline(base, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(base, indent=2))


def load_baseline(path):
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------- one batch
def evaluate_batch(sc, base, X, y=None, name="", cfg=None):
    """Return (summary_row, feature_psi_rows, alerts) for one batch of loans."""
    th = thresholds(cfg)
    scores, pdh = sc.score(X), sc.predict_pd(X)
    s = base["score"]
    row = {"batch": name, "loans": int(len(X)), "mean_score": float(np.mean(scores)),
           "mean_pd": float(np.mean(pdh))}
    row["score_psi"] = psi_from_shares(s["shares"], _numeric_shares(scores, s["edges"]))

    bands = pd.Series(sc.band(scores)).value_counts(normalize=True)
    now = [float(bands.get(b, 0.0)) for b in sc.band_names]
    row["band_psi"] = psi_from_shares([base["band_shares"][b] for b in sc.band_names], now)
    row["share_riskiest_band"] = now[0]

    frows = []
    for c, b in base["features"].items():
        if b["type"] == "numeric":
            cur = _numeric_shares(pd.to_numeric(X[c], errors="coerce").to_numpy(), b["edges"])
        else:
            cur = _cat_shares(X[c], b["categories"])
        v = psi_from_shares(b["shares"], cur)
        frows.append({"batch": name, "feature": c, "psi": v, "status": psi_status(v, th),
                      "in_model": c in base["selected_features"]})
    fdf = pd.DataFrame(frows)
    model_f = fdf[fdf.in_model] if fdf.in_model.any() else fdf
    top = model_f.sort_values("psi", ascending=False).iloc[0]
    row["max_feature_psi"], row["max_feature_name"] = float(top.psi), top.feature

    perf = base.get("performance")
    row.update({"auc": np.nan, "ks": np.nan, "bad_rate": np.nan, "calibration_ratio": np.nan, "auc_drop": np.nan})
    if y is not None:
        y = np.asarray(y)
        if len(y) >= th["min_batch_loans"] and y.sum() >= th["min_bads_for_auc"] and y.sum() < len(y):
            m = ranking_metrics(y, pdh)
            row.update({"auc": m["auc"], "ks": m["ks"], "bad_rate": m["bad_rate"],
                        "calibration_ratio": m["bad_rate"] / max(m["mean_pd"], 1e-9)})
            if perf:
                row["auc_drop"] = perf["auc"] - m["auc"]

    alerts = _alerts(row, fdf, th, name)
    row["status"] = max((a["severity"] for a in alerts), key=SEVERITY.get, default="stable")
    return row, fdf, alerts


def _alerts(row, fdf, th, name):
    out = []

    def add(sev, metric, value, msg):
        out.append({"batch": name, "severity": sev, "metric": metric, "value": float(value), "message": msg})

    for metric, label in (("score_psi", "Score distribution"), ("band_psi", "Risk band mix")):
        st = psi_status(row[metric], th)
        if st != "stable":
            add(st, metric, row[metric], f"{label} PSI {row[metric]:.3f} ({st}).")
    for r in fdf.itertuples():
        if r.in_model and r.status != "stable":
            add(r.status, f"psi:{r.feature}", r.psi, f"Feature {r.feature} PSI {r.psi:.3f} ({r.status}).")
    if not np.isnan(row["auc_drop"]):
        if row["auc_drop"] >= th["auc_drop_alert"]:
            add("alert", "auc_drop", row["auc_drop"], f"AUC fell {row['auc_drop']:.3f} versus baseline.")
        elif row["auc_drop"] >= th["auc_drop_watch"]:
            add("watch", "auc_drop", row["auc_drop"], f"AUC fell {row['auc_drop']:.3f} versus baseline.")
    if not np.isnan(row["calibration_ratio"]):
        gap = abs(row["calibration_ratio"] - 1)
        sev = "alert" if gap >= th["calib_alert"] else "watch" if gap >= th["calib_watch"] else None
        if sev:
            add(sev, "calibration", row["calibration_ratio"],
                f"Actual bad rate is {row['calibration_ratio']:.2f}x the predicted PD.")
    return out


# ----------------------------------------------------------------------------- many batches
def load_batch(path, outcomes_dir, cfg):
    """Read one batch file, apply the SAME cleaning as training, attach outcomes if they exist."""
    path = Path(path)
    df = load_raw(path).drop_duplicates("LoanSeqNum")
    df = prepare(df, cfg, with_targets=False)
    y = None
    ofile = Path(outcomes_dir) / path.name.replace("loans_", "outcomes_") if outcomes_dir else None
    if ofile is not None and ofile.exists():
        oc = pd.read_csv(ofile, dtype={"LoanSeqNum": str})[["LoanSeqNum", "MonthsDelinquent"]]
        df = df.merge(oc.drop_duplicates("LoanSeqNum"), on="LoanSeqNum", how="left")
        df = df[df["MonthsDelinquent"].notna()]
        y = add_targets(df, cfg)[cfg["target"]["primary"]].to_numpy()
    return df, y


def run_monitoring(sc, base, cfg, batches_dir, outcomes_dir, out_dir):
    from .config import model_inputs
    files = sorted(Path(batches_dir).glob("loans_*.csv"))
    if not files:
        raise FileNotFoundError(f"No loans_*.csv files in {batches_dir}")
    cols = model_inputs(cfg)
    rows, frows, alerts = [], [], []
    for f in files:
        df, y = load_batch(f, outcomes_dir, cfg)
        r, fdf, al = evaluate_batch(sc, base, df[cols], y, f.stem.replace("loans_", ""), cfg)
        rows.append(r), frows.append(fdf), alerts.extend(al)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    feat = pd.concat(frows, ignore_index=True)
    summary.to_csv(out / "batch_monitoring.csv", index=False)
    feat.to_csv(out / "feature_psi.csv", index=False)
    (out / "alerts.json").write_text(json.dumps(alerts, indent=2))
    plot_trend(summary, base, thresholds(cfg), out / "monitoring_trend.png")
    return summary, feat, alerts


def plot_trend(summary, base, th, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = summary["batch"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    a = ax[0, 0]
    a.plot(x, summary.score_psi, marker="o", label="score PSI")
    a.plot(x, summary.max_feature_psi, marker="s", label="largest feature PSI")
    a.axhline(th["psi_watch"], color="orange", ls="--", lw=1)
    a.axhline(th["psi_alert"], color="red", ls="--", lw=1)
    a.set_yscale("symlog", linthresh=0.1)
    a.set_title("Population stability (PSI, log scale): 0.10 watch, 0.25 alert")
    a.legend()
    a = ax[0, 1]
    a.plot(x, summary.auc, marker="o", label="AUC")
    if base.get("performance"):
        a.axhline(base["performance"]["auc"], color="gray", ls="--", label="baseline AUC")
    a.set_title("Ranking power (AUC) on loans with known outcomes")
    a.legend()
    a = ax[1, 0]
    a.plot(x, summary.mean_pd, marker="o", label="predicted PD (average)")
    a.plot(x, summary.bad_rate, marker="s", label="actual bad rate")
    a.set_title("Calibration: predicted vs actual")
    a.legend()
    a = ax[1, 1]
    a.plot(x, summary.share_riskiest_band, marker="o", color="firebrick")
    a.axhline(0.10, color="gray", ls="--", lw=1)
    a.set_title("Share of loans in the riskiest band (baseline 10%)")
    for k in ax.ravel():
        k.tick_params(axis="x", rotation=90, labelsize=7)
        k.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def first_breach(summary, column, limit):
    """Name of the first batch where `column` reaches `limit`, else None."""
    hit = summary[summary[column] >= limit]
    return None if hit.empty else str(hit.iloc[0]["batch"])
