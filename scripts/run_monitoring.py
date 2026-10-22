#!/usr/bin/env python3
"""Score every simulated monthly batch and write the monitoring tables and chart.

    python scripts/run_monitoring.py

Reads  artifacts/<target>/scorecard.joblib and baseline.json, data/sim/batches, data/sim/outcomes
Writes monitoring/batch_monitoring.csv, feature_psi.csv, alerts.json, monitoring_trend.png
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from credit_risk.config import load_config  # noqa: E402
from credit_risk.monitor import first_breach, load_baseline, run_monitoring, thresholds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["primary", "secondary"], default="primary")
    ap.add_argument("--batches", default="data/sim/batches")
    ap.add_argument("--outcomes", default="data/sim/outcomes")
    ap.add_argument("--out", default="monitoring")
    a = ap.parse_args()
    cfg = load_config()
    folder = Path(cfg["paths"]["artifacts"]) / cfg["target"][a.target]
    sc = joblib.load(folder / "scorecard.joblib")
    base = load_baseline(folder / "baseline.json")
    summary, feat, alerts = run_monitoring(sc, base, cfg, a.batches, a.outcomes, a.out)
    th = thresholds(cfg)
    pd.set_option("display.width", 220)
    show = summary[["batch", "loans", "score_psi", "max_feature_psi", "max_feature_name", "auc", "bad_rate", "mean_pd",
                    "calibration_ratio", "status"]]
    print(show.round(3).to_string(index=False))
    print()
    print("First batch with score PSI >= watch :", first_breach(summary, "score_psi", th["psi_watch"]))
    print("First batch with score PSI >= alert :", first_breach(summary, "score_psi", th["psi_alert"]))
    print("Baseline AUC:", round(base["performance"]["auc"], 3), "| latest AUC:", round(summary.auc.dropna().iloc[-1], 3))
    print(f"Wrote {len(alerts)} alerts and tables to {a.out}/")


if __name__ == "__main__":
    main()
