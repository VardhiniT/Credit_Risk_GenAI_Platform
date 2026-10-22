#!/usr/bin/env python3
"""Freeze the monitoring baseline from the real development data.

    python scripts/build_baseline.py --raw data/raw/FreddieMac.csv

Uses the same train/test split as training: the TRAIN part defines the PSI bins,
the TEST part defines the baseline AUC/KS. Writes artifacts/<target>/baseline.json.
"""
import argparse
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from credit_risk import data  # noqa: E402
from credit_risk.config import load_config, model_inputs  # noqa: E402
from credit_risk.monitor import build_baseline, save_baseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--target", choices=["primary", "secondary"], default="primary")
    ap.add_argument("--artifacts", default=None, help="folder holding scorecard.joblib (default artifacts/<target>)")
    a = ap.parse_args()
    cfg = load_config()
    target = cfg["target"][a.target]
    folder = Path(a.artifacts or Path(cfg["paths"]["artifacts"]) / target)
    sc = joblib.load(folder / "scorecard.joblib")
    df = data.prepare(data.load_raw(a.raw or cfg["paths"]["raw"]), cfg)
    y, cols = df[target].to_numpy(), model_inputs(cfg)
    X_tr, X_te, y_tr, y_te = train_test_split(df[cols], y, test_size=cfg["split"]["test_size"],
                                              random_state=cfg["split"]["random_state"], stratify=y)
    base = build_baseline(sc, cfg, X_tr, X_te, y_te)
    save_baseline(base, folder / "baseline.json")
    p = base["performance"]
    print(f"Baseline saved to {folder / 'baseline.json'}")
    print(f"Reference loans {base['n_reference']:,} | baseline test AUC {p['auc']:.3f}  KS {p['ks']:.3f}  bad rate {p['bad_rate']:.3f}")


if __name__ == "__main__":
    main()
