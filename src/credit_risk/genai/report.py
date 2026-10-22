"""Automated monitoring report (markdown). Tables and recommendations come from code; the narrative
paragraph goes through the same governed assistant (LLM if available, template otherwise)."""
import datetime as dt
from pathlib import Path

import pandas as pd

from ..monitor import thresholds
from . import scenarios as sc_mod
from .governance import LABELS, GovernanceError, validate_fields


def _clean(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return v


def _table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and pd.isna(v)) else str(_clean(v)) for v in r) + " |")
    return "\n".join(lines)


def build_report(service, assistant, out_path):
    summary, feat = service._monitoring()
    if summary is None:
        raise FileNotFoundError("No monitoring output found. Run `make monitor` first.")
    th = thresholds(service.cfg)
    facts = service.drift_status()
    narrative = assistant.ask("Summarise the drift and monitoring status of the model.")["answer"]
    last = summary.iloc[-1]

    recs = []
    if last["score_psi"] >= th["psi_alert"]:
        recs.append("Score distribution has shifted materially (PSI above 0.25): investigate the population change before relying on scores.")
    elif last["score_psi"] >= th["psi_watch"]:
        recs.append("Score distribution is drifting (PSI above 0.10): keep the model on watch and review monthly.")
    if "auc_drop" in summary and summary["auc_drop"].iloc[-1] >= th["auc_drop_alert"]:
        recs.append("AUC has fallen by more than 0.05 versus baseline: schedule a model redevelopment or recalibration review.")
    if abs(last["calibration_ratio"] - 1) >= th["calib_alert"] if pd.notna(last["calibration_ratio"]) else False:
        recs.append("Predicted PDs no longer match actual bad rates: recalibrate before using PDs for pricing or provisioning.")
    if not recs:
        recs.append("No action needed. Continue routine monthly monitoring.")

    show = summary[["batch", "loans", "score_psi", "max_feature_psi", "auc", "bad_rate", "mean_pd", "status"]].copy()
    show = show.round({"score_psi": 3, "max_feature_psi": 3, "auc": 3, "bad_rate": 3, "mean_pd": 3})
    last_feat = feat[(feat.batch == last["batch"])].sort_values("psi", ascending=False).head(8)
    last_feat = last_feat[["feature", "psi", "status", "in_model"]].round({"psi": 3})

    md = f"""# Model monitoring report

Generated {dt.date.today()} from `{service.monitoring_dir}/`. Batches covered: {facts['batches']} (latest: {facts['latest_batch']}).

> The monthly batches are **simulated** with controlled drift. This report shows that the monitoring and reporting
> machinery works; it is not evidence of drift in a real portfolio.

## Summary

{narrative}

## Recommended actions

""" + "\n".join(f"- {r}" for r in recs) + f"""

## Thresholds used

PSI: below {th['psi_watch']} stable, {th['psi_watch']} to {th['psi_alert']} watch, above {th['psi_alert']} alert.
AUC drop: watch at {th['auc_drop_watch']}, alert at {th['auc_drop_alert']}.
Calibration (actual bad rate / predicted PD): watch beyond {1 - th['calib_watch']:.2f} to {1 + th['calib_watch']:.2f}, alert beyond {1 - th['calib_alert']:.2f} to {1 + th['calib_alert']:.2f}.

## Feature stability, latest batch ({last['batch']})

{_table(last_feat)}

## All batches

{_table(show)}
"""
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(md)
    return md


# ----------------------------------------------------------------------------- loan-level note
def _scenario_specs(loan):
    cs = float(loan["CreditScore"])
    return [("Credit score +40", [{"field": "CreditScore", "op": "set", "value": min(850.0, cs + 40)}]),
            ("LTV down 10 points", [{"field": "LTV", "op": "add", "value": -10}]),
            ("DTI down 5 points", [{"field": "DTI", "op": "add", "value": -5}]),
            ("Loan amount +10%, property value unchanged", [{"field": "OrigUPB", "op": "pct", "value": 10}]),
            ("Interest rate +1 point", [{"field": "OrigInterestRate", "op": "add", "value": 1}]),
            (sc_mod.STRESS_PRESETS["downturn"][0], sc_mod.STRESS_PRESETS["downturn"][1])]


def build_loan_report(service, assistant, loan, out_dir="reports"):
    """Write a one-loan decision-support note (markdown). Returns (path, facts). No personal data is involved:
    only the model inputs appear, and the note is labelled as a model estimate, not a credit decision."""
    loan = validate_fields(loan)
    f = service.score_loan(loan)
    narrative, _, _ = assistant.explain_facts("explain_score", "Explain this loan's risk score.", f)

    rows = []
    for label, specs in _scenario_specs(loan):
        try:
            w = service.what_if(loan, specs, label)
            rows.append([label, w["after_score"], f"{w['score_change']:+d}", f"{w['after_pd_percent']}%", w["after_band"]])
        except GovernanceError:
            rows.append([label, "outside model range", "", "", ""])
    takes = service.what_it_takes(loan, {})
    health = service.drift_status()

    reasons = pd.DataFrame([{"factor": r["factor"], "loan value": r["value"], "portfolio median": r.get("portfolio_median", ""),
                             "points lost": r["points_lost"]} for r in f["reasons"]])
    inputs = pd.DataFrame([{"input": LABELS.get(k, k), "value": v} for k, v in f["inputs"].items() if k != "annual_income"])

    if takes.get("already_best"):
        takes_md = f"The loan is already in the safest band ({takes['current_band']})."
    elif takes.get("levers"):
        lv = []
        for l in takes["levers"]:
            if l["reachable"]:
                extra = f" (about ${l['extra_down_payment']:,.0f} more down payment)" if "extra_down_payment" in l else ""
                lv.append([l["lever"], _clean(float(l["from"])), _clean(float(l["to"])), f"{l['change']:+g}{extra}", l["score_after"]])
            else:
                lv.append([l["lever"], "", "not reachable alone", "", ""])
        needs = f" It needs at least {takes['points_needed']} more points." if takes.get("points_needed") else ""
        takes_md = (f"Goal: {takes['goal']}.{needs} Each lever is moved on its own, everything else unchanged.\n\n"
                    + _table(pd.DataFrame(lv, columns=["lever", "from", "to", "change", "score after"])))
    else:
        takes_md = "No change needed."

    if health.get("available"):
        caveat = (f"Model status in the latest monitored batch ({health['latest_batch']}): **{health['status']}** "
                  f"(score PSI {health['score_psi']}). " +
                  ("Treat this score with extra caution until the model is reviewed." if health["status"] == "alert" else ""))
    else:
        caveat = "No monitoring results were available when this note was written."

    band_line = ""
    if "band_historical_bad_rate_percent" in f:
        band_line = f"\nLoans in band {f['risk_band']} had an observed bad rate of {f['band_historical_bad_rate_percent']}% on held-out data."
    md = f"""# Loan risk note

Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Model-based estimate for portfolio analysis. **Not a credit decision.**

## Summary

{narrative}

## Risk view

| | |
|---|---|
| Score | {f['score']} |
| Estimated probability of serious delinquency (3+ months late) | {f['pd_percent']}% |
| Risk band | {f['risk_band']} ({f['band_position']}) |
| Portfolio average PD | {f.get('portfolio_average_pd_percent', 'n/a')}% |
{band_line}

## Main reasons

{_table(reasons) if len(reasons) else "No single factor lowers the score noticeably."}

## Scenarios

{_table(pd.DataFrame(rows, columns=["scenario", "score", "change", "PD", "band"]))}

Scenario assumptions: loan amount changes keep the property value fixed (so LTV moves); stress cases are illustrative.

## What it would take to improve

{takes_md}

## Model reliability

{caveat}

## Loan inputs used

{_table(inputs)}

## Limits

One origination vintage (Q1 1999), random train/test split, no income or bureau history in the training data, drift
results on simulated batches, and no fairness metrics by segment yet. The score ranks risk; it does not decide lending.
"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"loan_note_{dt.datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(md)
    return path, {"summary_facts": f, "scenarios": len(rows), "path": str(path)}
