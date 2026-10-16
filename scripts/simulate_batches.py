#!/usr/bin/env python3
"""
Simulate future monthly loan batches (with drift) from your REAL Freddie Mac data.

WHY: the real file is a single vintage (Q1 1999), so it cannot show new data arriving
or model drift. This script creates what production would look like:

  * monthly batches of NEW loans    -> data/sim/batches/loans_YYYY_MM.csv     (features only)
  * outcomes that arrive later      -> data/sim/outcomes/outcomes_YYYY_MM.csv (EverDelinquent, MonthsDelinquent)
  * documented drift schedule       -> data/sim/_manifest.json
  * per-batch sanity statistics     -> data/sim/_batch_summary.csv
  * tiny unit-test fixture          -> tests/fixtures/loans_sample.csv

HOW OUTCOMES ARE MADE (so you can explain it):
  1. Two small "generator" models are fitted on the REAL loans (origination features only):
        P(ever delinquent | x)                      and
        P(3+ months delinquent | x, ever delinquent)
     They are NOT your scorecard. They only make simulated outcomes realistic.
  2. For each new loan: draw ever-delinquent, then draw serious (3+ months) among those.
     So serious always implies ever, exactly like the real data.
  3. MonthsDelinquent is then drawn from the real distribution (1-2 months for mild cases,
     3+ for serious cases, capped at 24 = the simulated observation window).
  4. Drift (controlled, documented): applicant quality worsens over time (lower scores, higher
     DTI/LTV/rates, mix shift to growth states) AND the signal fades while default risk rises
     ("concept drift"). The final summary shows both.
  5. Dirty data (duplicates, missing DTI, credit score 0) is injected at a small rate.

Everything here is SIMULATED. Rows carry DataSource='SIMULATED' and LoanSeqNum starting with 'S'.

Usage (from the project root):
  python scripts/simulate_batches.py --real data/raw/FreddieMac.csv
  python scripts/simulate_batches.py --real data/raw/FreddieMac.csv --drift-strength 0   # no drift
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Generator inputs are origination-time columns only (never outcomes, never MonthsInRepayment).
NUM = ["CreditScore", "DTI", "LTV", "OCLTV", "OrigInterestRate", "OrigUPB", "MIP", "Units", "OrigLoanTerm"]
CAT = ["FirstTimeHomebuyer", "Occupancy", "Channel", "PropertyType", "LoanPurpose",
       "NumBorrowers", "PropertyState"]
OUTCOME_COLS = ["EverDelinquent", "MonthsDelinquent", "MonthsInRepayment"]
GROWTH_STATES = ["CA", "FL", "TX", "AZ", "NV"]   # mix shifts toward these over time
WINDOW = 24                                      # simulated observation window in months


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def prep(df, score_median):
    d = df[NUM + CAT].copy()
    d["CreditScore"] = d["CreditScore"].replace(0, np.nan).fillna(score_median)
    d["DTI"] = d["DTI"].fillna(d["DTI"].median() if d["DTI"].notna().any() else 35)
    for c in CAT:
        d[c] = d[c].astype(str)
    return d


def _make_model():
    pre = ColumnTransformer([
        ("n", StandardScaler(), NUM),
        ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


def fit_generators(real):
    """Return the ever-delinquent model, the serious-given-ever model, and their mean log-odds."""
    med = float(real.loc[real.CreditScore > 0, "CreditScore"].median())
    X = prep(real, med)
    ever = (real["MonthsDelinquent"] >= 1).astype(int)
    serious = (real["MonthsDelinquent"] >= 3).astype(int)
    m_e = _make_model().fit(X, ever)
    sub = ever == 1
    m_q = _make_model().fit(X[sub], serious[sub])
    ze = float(logit(m_e.predict_proba(X)[:, 1]).mean())
    zq = float(logit(m_q.predict_proba(X)[:, 1]).mean())
    md = real.loc[real["MonthsDelinquent"] >= 1, "MonthsDelinquent"].to_numpy()
    mild = md[md < 3]
    dist = {"p_one_month_among_mild": float((mild == 1).mean()),
            "serious_values": np.minimum(md[md >= 3], WINDOW)}
    return m_e, m_q, med, ze, zq, dist


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def yyyymm_add(yyyymm, months):
    y, m = divmod(yyyymm, 100)
    t = y * 12 + (m - 1) + months
    return (t // 12) * 100 + (t % 12) + 1


def drift_z(z, zbar, sp, cfg):
    """Fading signal (shrink toward the mean) plus rising overall risk."""
    return zbar + (z - zbar) * (1 - cfg["decay"] * sp) + cfg["risk_up"] * sp


def simulate_month(real, rng, n, k, n_months, s, start_yyyymm, gens, cfg):
    m_e, m_q, med, ze_bar, zq_bar, dist = gens
    p = k / (n_months - 1) if n_months > 1 else 0.0
    sp = s * p
    month = yyyymm_add(start_yyyymm, k)
    year, mm = divmod(month, 100)

    w = np.where(real.PropertyState.isin(GROWTH_STATES), 1 + cfg["state_shift"] * sp, 1.0)
    idx = rng.choice(len(real), size=n, p=w / w.sum())
    b = real.iloc[idx].reset_index(drop=True).drop(columns=OUTCOME_COLS)

    # --- drift in applicant quality ---------------------------------------
    cs = b["CreditScore"].to_numpy().astype(float)
    known = cs > 0                                           # 0 = unknown, keep as is
    cs[known] = np.clip(cs[known] + cfg["score_shift"] * sp + rng.normal(0, 5, known.sum()), 300, 850)
    b["CreditScore"] = np.round(cs).astype(int)

    dti = b["DTI"].to_numpy().astype(float)
    kn = dti > 0
    dti[kn] = np.clip(dti[kn] + cfg["dti_shift"] * sp + rng.normal(0, 1.5, kn.sum()), 1, 65)
    b["DTI"] = np.round(dti).astype(int)

    d_ltv = np.round(cfg["ltv_shift"] * sp + rng.normal(0, 1.0, n))
    ltv = np.clip(b["LTV"].to_numpy() + d_ltv, 10, 100)
    b["LTV"] = ltv.astype(int)
    b["OCLTV"] = np.maximum(b["OCLTV"].to_numpy() + d_ltv, ltv).clip(10, 125).astype(int)

    rate = b["OrigInterestRate"].to_numpy() + cfg["rate_shift"] * sp + rng.normal(0, 0.12, n)
    b["OrigInterestRate"] = np.clip(np.round(rate * 8) / 8, 2.0, 14.0)

    upb = b["OrigUPB"].to_numpy() * (1 + cfg["upb_growth"] * sp) * rng.lognormal(0, 0.03, n)
    b["OrigUPB"] = (np.round(upb / 1000) * 1000).astype(int).clip(10_000, None)

    # --- identifiers and dates --------------------------------------------
    seq = k * 100_000 + np.arange(n)
    b["LoanSeqNum"] = [f"S{year % 100:02d}Q{(mm - 1) // 3 + 1}{x:07d}" for x in seq]
    b["FirstPaymentDate"] = month
    b["MaturityDate"] = [yyyymm_add(month, int(t) - 1) for t in b["OrigLoanTerm"]]
    b["DataSource"] = "SIMULATED"

    # --- outcomes: ever delinquent, then serious among those ---------------
    Xp = prep(b, med)
    pe0 = m_e.predict_proba(Xp)[:, 1]
    pq0 = m_q.predict_proba(Xp)[:, 1]
    pe = sigmoid(drift_z(logit(pe0), ze_bar, sp, cfg))
    pq = sigmoid(drift_z(logit(pq0), zq_bar, sp, cfg))
    y_ever = (rng.random(n) < pe).astype(int)
    y_serious = y_ever * (rng.random(n) < pq).astype(int)

    months = np.zeros(n, dtype=int)
    mild = (y_ever == 1) & (y_serious == 0)
    months[mild] = np.where(rng.random(mild.sum()) < dist["p_one_month_among_mild"], 1, 2)
    ser = y_serious == 1
    months[ser] = rng.choice(dist["serious_values"], size=ser.sum())
    outcomes = pd.DataFrame({"LoanSeqNum": b["LoanSeqNum"], "EverDelinquent": y_ever,
                             "MonthsDelinquent": months})

    base_serious = pe0 * pq0     # baseline (undrifted) chance of serious delinquency
    stats = {
        "batch": f"{year}_{mm:02d}", "loans": n,
        "mean_credit_score": float(b.loc[b.CreditScore > 0, "CreditScore"].mean()),
        "mean_dti": float(b.loc[b.DTI > 0, "DTI"].mean()),
        "mean_ltv": float(b["LTV"].mean()),
        "mean_rate": float(b["OrigInterestRate"].mean()),
        "ever_rate": float(y_ever.mean()),
        "serious_rate": float(y_serious.mean()),
        "auc_of_baseline_generator": float(roc_auc_score(y_serious, base_serious)),
        "psi_credit_score_vs_real": psi(real.loc[real.CreditScore > 0, "CreditScore"].to_numpy(),
                                        b.loc[b.CreditScore > 0, "CreditScore"].to_numpy()),
    }
    return b, outcomes, stats, year, mm


def inject_dirt(b, rng, rate):
    """Small amount of realistic mess so pipeline and dbt tests have work to do."""
    if rate <= 0:
        return b
    b = b.copy()
    n = len(b)
    b["DTI"] = b["DTI"].astype("Int64")
    b.loc[rng.random(n) < rate, "DTI"] = pd.NA                     # missing DTI
    b.loc[rng.random(n) < rate / 2, "CreditScore"] = 0              # unknown-score sentinel
    dups = b.sample(frac=rate / 2, random_state=int(rng.integers(0, 10**6)))
    return pd.concat([b, dups], ignore_index=True)                  # duplicate rows


def make_fixture(real, rng, path):
    """100 real rows incl. edge cases, for unit tests."""
    f = real.sample(100, random_state=int(rng.integers(0, 10**6))).reset_index(drop=True)
    f.loc[0, "CreditScore"] = 0                 # unknown score
    f.loc[1, "SellerName"] = np.nan             # missing seller
    f.loc[2, "MSA"] = "99999"                   # category never seen in training
    f.loc[3, "DTI"] = 0                         # DTI = 0 case
    f.loc[4, ["EverDelinquent", "MonthsDelinquent"]] = [1, 2]    # boundary: 2 months -> NOT serious
    f.loc[5, ["EverDelinquent", "MonthsDelinquent"]] = [1, 3]    # boundary: 3 months -> serious
    f.loc[6, ["EverDelinquent", "MonthsDelinquent"]] = [0, 0]
    f.loc[7, "FirstTimeHomebuyer"] = "X"        # unknown code
    path.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="path to the raw FreddieMac.csv")
    ap.add_argument("--out", default="data/sim")
    ap.add_argument("--fixture", default="tests/fixtures/loans_sample.csv")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--per-batch", type=int, default=10_000)
    ap.add_argument("--start", default="2000-01", help="first simulated batch month, YYYY-MM")
    ap.add_argument("--drift-strength", type=float, default=1.0, help="0 = no drift, 1 = default, 2 = severe")
    ap.add_argument("--dirty-rate", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = dict(score_shift=-30, dti_shift=6, ltv_shift=5, rate_shift=1.0, upb_growth=0.20,
               state_shift=1.5, decay=0.30, risk_up=0.45)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "outcomes").mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(a.real, dtype={"PostalCode": str, "MSA": str, "LoanSeqNum": str}, low_memory=False)
    print(f"Real data: {len(real):,} loans | fitting generator models ...")
    gens = fit_generators(real)
    sy, sm = map(int, a.start.split("-"))
    start = sy * 100 + sm

    rows = []
    for k in range(a.months):
        b, outc, stats, year, mm = simulate_month(real, rng, a.per_batch, k, a.months,
                                                  a.drift_strength, start, gens, cfg)
        b = inject_dirt(b, rng, a.dirty_rate)
        b.to_csv(out / "batches" / f"loans_{year}_{mm:02d}.csv", index=False)
        outc.to_csv(out / "outcomes" / f"outcomes_{year}_{mm:02d}.csv", index=False)
        rows.append(stats)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "_batch_summary.csv", index=False)
    (out / "_manifest.json").write_text(json.dumps({
        "note": "SIMULATED data derived from real Freddie Mac Q1-1999 loans. Not real borrowers.",
        "seed": a.seed, "months": a.months, "loans_per_batch": a.per_batch, "start": a.start,
        "drift_strength": a.drift_strength, "dirty_rate": a.dirty_rate, "observation_window_months": WINDOW,
        "drift_schedule_at_strength_1_final_month": cfg,
    }, indent=2))
    make_fixture(real, rng, Path(a.fixture))

    print(f"\nWrote {a.months} monthly batches to {out}/  (+ outcomes, manifest, fixture)\n")
    show = summary.iloc[[0, a.months // 4, a.months // 2, 3 * a.months // 4, a.months - 1]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
