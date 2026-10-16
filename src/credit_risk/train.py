"""Train and evaluate the scorecard.

    python -m credit_risk.train --raw data/raw/FreddieMac.csv
    python -m credit_risk.train --raw data/raw/FreddieMac.csv --target secondary   # sensitivity check

Outputs go to artifacts/<target_name>/ : scorecard.joblib, metrics.json, iv_table.csv,
woe_bins.csv, coefficients.csv, points.csv, deciles_test.csv, bands_test.csv, model_card.md
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import data
from .config import load_config, model_inputs
from .evaluate import band_table, decile_table, is_monotone_decreasing_risk, ranking_metrics
from .scorecard import Scorecard


def cross_val_auc(df, y, cfg, folds=5):
    """Refit binning AND model inside every fold, so binning never sees the validation part."""
    cols = model_inputs(cfg)
    aucs = []
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=cfg["split"]["random_state"]).split(df, y):
        sc = Scorecard(cfg).fit(df.iloc[tr][cols], y[tr])
        aucs.append(ranking_metrics(y[va], sc.predict_pd(df.iloc[va][cols]))["auc"])
    return [float(a) for a in aucs]


def run(raw_path, cfg, target_key="primary", out_root=None, cv_folds=5):
    target = cfg["target"][target_key]
    df = data.prepare(data.load_raw(raw_path), cfg)
    y = df[target].to_numpy()
    cols = model_inputs(cfg)

    X_tr, X_te, y_tr, y_te = train_test_split(
        df[cols], y, test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"], stratify=y)

    sc = Scorecard(cfg).fit(X_tr, y_tr)

    pd_tr, pd_te = sc.predict_pd(X_tr), sc.predict_pd(X_te)
    s_tr, s_te = sc.score(X_tr), sc.score(X_te)
    metrics = {"target": target, "rows": int(len(df)),
               "train": ranking_metrics(y_tr, pd_tr), "test": ranking_metrics(y_te, pd_te)}
    if cv_folds:
        cv = cross_val_auc(df, y, cfg, cv_folds)
        metrics["cv_auc"] = {"folds": cv, "mean": float(np.mean(cv)), "std": float(np.std(cv))}

    deciles = decile_table(y_te, pd_te)
    b_te = sc.band(s_te)
    bands = band_table(y_te, b_te, sc.band_names)
    metrics["bands_monotone"] = is_monotone_decreasing_risk(bands)
    metrics["selected_features"] = sc.selected_
    metrics["dropped_features"] = sc.dropped_
    metrics["band_cutoffs"] = sc.band_cutoffs_
    metrics["max_calibration_gap_decile"] = float((deciles["mean_pd"] - deciles["actual_rate"]).abs().max())

    out = Path(out_root or cfg["paths"]["artifacts"]) / target
    out.mkdir(parents=True, exist_ok=True)
    sc.save(out / "scorecard.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sc.binner.iv_table().to_csv(out / "iv_table.csv", index=False)
    sc.binner.bins_table().to_csv(out / "woe_bins.csv", index=False)
    sc.coef_table().to_csv(out / "coefficients.csv", index=False)
    sc.points_table().to_csv(out / "points.csv", index=False)
    deciles.to_csv(out / "deciles_test.csv", index=False)
    bands.to_csv(out / "bands_test.csv", index=False)
    (out / "model_card.md").write_text(model_card(cfg, sc, metrics, bands))
    return sc, metrics, deciles, bands, out


def model_card(cfg, sc, m, bands):
    t, te, tr = m["target"], m["test"], m["train"]
    excl = "\n".join(f"- `{c}`" for c in cfg["forbidden_features"])
    drops = "\n".join(f"- `{k}`: {v}" for k, v in m["dropped_features"].items()) or "- none"
    cv = m.get("cv_auc")
    cv_line = f"\n- 5-fold CV AUC: {cv['mean']:.3f} (std {cv['std']:.3f})" if cv else ""
    return f"""# Model card: credit risk scorecard ({t})

Generated {dt.date.today()} by `credit_risk.train`. Do not edit by hand; re-run training.

## Purpose
Rank mortgage loans by probability of `{t}` using only information known at approval.
Intended for monitoring and risk-based segmentation in a portfolio project. Not for real lending decisions.

## Target
`{t}`: {'MonthsDelinquent >= ' + str(cfg['target']['serious_months']) if t == cfg['target']['primary'] else 'MonthsDelinquent >= 1'}.
Observed bad rate: {te['bad_rate']:.1%} (test set).

## Data
Real Freddie Mac loans, single origination vintage (Q1 1999), {m['rows']:,} loans. Random stratified split
({int(cfg['split']['test_size']*100)}% test). Simulated batches are used only to test the pipeline and monitoring, never to train.

## Method
WOE binning (monotone trend for numeric variables, min bin {cfg['woe']['min_bin_pct']:.0%}), IV filter >= {cfg['woe']['min_iv']},
correlation filter, unpenalised logistic regression, sign check (all coefficients must agree with WOE direction).
Score = offset + factor * ln((1-PD)/PD); base {cfg['scorecard']['base_score']} points at {cfg['scorecard']['base_odds']}:1 odds, PDO {cfg['scorecard']['pdo']}.

## Performance
| Set | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| Train | {tr['auc']:.3f} | {tr['gini']:.3f} | {tr['ks']:.3f} | {tr['brier']:.4f} |
| Test | {te['auc']:.3f} | {te['gini']:.3f} | {te['ks']:.3f} | {te['brier']:.4f} |
{cv_line}
- Mean predicted PD vs actual (test): {te['mean_pd']:.3f} vs {te['bad_rate']:.3f}
- Largest decile calibration gap: {m['max_calibration_gap_decile']:.3f}
- Risk bands monotone (bad rate falls from riskiest to safest band): {m['bands_monotone']}

## Features
Used ({len(m['selected_features'])}): {', '.join('`'+f+'`' for f in m['selected_features'])}

Dropped during selection:
{drops}

Excluded by design (leakage, redundancy, or fair-lending proxy):
{excl}

## Limitations
- One vintage: out-of-time validation is not possible on the real data. Drift is demonstrated only on simulated batches.
- `MonthsDelinquent` counts total late months (not necessarily consecutive) and loans observed longer have more time to accumulate them.
- No income, employment, bureau history or macroeconomic variables are available.
- Geography and seller are kept as monitoring segments only, to limit fair-lending proxy risk. Fairness metrics by segment are not yet computed.
- A random split on a single vintage is optimistic compared with out-of-time results.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw", default=None, help="override paths.raw in the config")
    ap.add_argument("--target", choices=["primary", "secondary"], default="primary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cv", type=int, default=5, help="CV folds, 0 to skip")
    a = ap.parse_args()
    cfg = load_config(a.config)
    sc, m, deciles, bands, out = run(a.raw or cfg["paths"]["raw"], cfg, a.target, a.out, a.cv)
    pd.set_option("display.width", 200)
    print(f"Target: {m['target']}   rows: {m['rows']:,}")
    for k in ("train", "test"):
        r = m[k]
        print(f"{k:5s} AUC {r['auc']:.3f}  Gini {r['gini']:.3f}  KS {r['ks']:.3f}  bad rate {r['bad_rate']:.3f}  mean PD {r['mean_pd']:.3f}")
    if "cv_auc" in m:
        print(f"CV AUC {m['cv_auc']['mean']:.3f} +/- {m['cv_auc']['std']:.3f}")
    print("Selected:", ", ".join(m["selected_features"]))
    for k, v in m["dropped_features"].items():
        print(f"Dropped  {k}: {v}")
    print("\nRisk bands (test):\n", bands.round(3).to_string(index=False))
    print("\nBands monotone:", m["bands_monotone"], "| artifacts ->", out)


if __name__ == "__main__":
    main()
