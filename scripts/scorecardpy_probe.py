"""Trial run: does scorecardpy work on our data, and what does it give us?

Changes nothing in the project. Run:  python scripts/scorecardpy_probe.py
Uses the same cleaning (data.py), the same split and the same score scaling as our own model.
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from credit_risk import data
from credit_risk.config import load_config, model_inputs
from credit_risk.evaluate import ranking_metrics

print("pandas", pd.__version__, "| numpy", np.__version__, "| scorecardpy", sc.__version__)

cfg = load_config()
target = cfg["target"]["primary"]
df = data.prepare(data.load_raw(cfg["paths"]["raw"]), cfg)
cols = model_inputs(cfg)

train, test = train_test_split(
    df[cols + [target]], test_size=cfg["split"]["test_size"],
    random_state=cfg["split"]["random_state"], stratify=df[target])
print(f"train {len(train):,} rows, test {len(test):,} rows, bad rate {train[target].mean():.3%}")

# 1) binning + WOE + IV (scorecardpy does the binning)
t0 = time.time()
bins = sc.woebin(train, y=target, x=cols, bin_num_limit=8, count_distr_limit=0.05,
                 method="tree", no_cores=1, print_step=0)
print(f"\nwoebin finished in {time.time() - t0:.1f}s")

iv = pd.Series({v: float(b["total_iv"].iloc[0]) for v, b in bins.items()}).sort_values(ascending=False)
print("\nInformation value per feature (ours dropped anything below 0.02):")
print(iv.round(4).to_string())

print("\nBins for CreditScore (look at the sign of woe: positive = riskier here?):")
print(bins["CreditScore"][["bin", "count_distr", "good", "bad", "badprob", "woe", "bin_iv"]].round(4).to_string(index=False))

# 2) keep features with IV >= 0.02, replace values with WOE, fit logistic regression
keep = [v for v in iv.index if iv[v] >= cfg["woe"]["min_iv"]]
train_woe = sc.woebin_ply(train[keep + [target]], {k: bins[k] for k in keep}, no_cores=1, print_step=0)
test_woe = sc.woebin_ply(test[keep + [target]], {k: bins[k] for k in keep}, no_cores=1, print_step=0)
xcols = [c for c in train_woe.columns if c != target]
lr = LogisticRegression(C=1e6, max_iter=2000).fit(train_woe[xcols], train[target])
print("\nCoefficients:")
print(pd.Series(lr.coef_[0], index=xcols).round(3).to_string())

# 3) points, with the same scaling as our own scorecard (600 at 50:1, 20 points double the odds)
s = cfg["scorecard"]
card = sc.scorecard({k: bins[k] for k in keep}, lr, xcols,
                    points0=s["base_score"], odds0=1 / s["base_odds"], pdo=s["pdo"])
test_score = sc.scorecard_ply(test[keep], card, only_total_score=True, print_step=0)["score"]
pd_test = lr.predict_proba(test_woe[xcols])[:, 1]

m = ranking_metrics(test[target].to_numpy(), pd_test)
print(f"\nTEST  AUC {m['auc']:.3f}  Gini {m['gini']:.3f}  KS {m['ks']:.3f}  "
      f"bad rate {m['bad_rate']:.3f}  mean PD {m['mean_pd']:.3f}")
print("(our own scorecard: AUC 0.773, Gini 0.545, KS 0.425)")
print(f"score range on test: {test_score.min():.0f} to {test_score.max():.0f}, mean {test_score.mean():.0f}")
print("\nPoints table for CreditScore:")
print(card["CreditScore"][["bin", "points"]].to_string(index=False))