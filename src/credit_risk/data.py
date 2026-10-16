"""Load, clean and label loan data.

The same cleaning rules are applied to real and simulated rows. Later, dbt staging models
implement the same rules in SQL, so keep this file and those models in sync.
"""
import numpy as np
import pandas as pd

TEXT_COLS_STRIP = True


def load_raw(path):
    df = pd.read_csv(path, dtype={"PostalCode": str, "MSA": str, "LoanSeqNum": str}, low_memory=False)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()
    return df


def clean(df, cfg):
    """Return a cleaned copy. Raw values are never modified in place."""
    df = df.copy()
    unk = cfg["cleaning"]["unknown_code"]

    # 'X' codes mean "unknown" in this dataset -> explicit UNKNOWN category
    for c in cfg["features"]["categorical"]:
        s = df[c].astype("string").str.strip()
        df[c] = s.where(s.notna() & (s != unk) & (s != ""), "UNKNOWN").astype(object)

    # Segment-only columns: same treatment, kept for monitoring and fairness
    for c in ("MSA", "SellerName"):
        if c in df.columns:
            s = df[c].astype("string").str.strip()
            df[c] = s.where(s.notna() & (s != unk) & (s != ""), "UNKNOWN").astype(object)

    # 0 means "unknown" for these columns -> NaN, plus an explicit flag
    flags = {"CreditScore": "cs_unknown", "DTI": "dti_unknown"}
    for c in cfg["cleaning"]["zero_means_unknown"]:
        v = pd.to_numeric(df[c], errors="coerce")
        df[flags[c]] = (v.fillna(0) == 0).astype(int)
        df[c] = v.where(v > 0)
    return df


def add_derived(df):
    df = df.copy()
    df["ltv_gt80"] = (pd.to_numeric(df["LTV"], errors="coerce") > 80).astype(int)
    if "FirstPaymentDate" in df.columns:
        fp = pd.to_numeric(df["FirstPaymentDate"], errors="coerce")
        year = (fp // 100).astype("Int64")
        month = (fp % 100).astype("Int64")
        q = ((month - 1) // 3 + 1).astype("Int64")
        df["vintage_quarter"] = (year.astype("string") + "Q" + q.astype("string")).astype(object)
    return df


def add_targets(df, cfg):
    """Create target columns from MonthsDelinquent. Outcome columns are never features."""
    t = cfg["target"]
    md = pd.to_numeric(df["MonthsDelinquent"], errors="coerce")
    df = df.copy()
    df[t["primary"]] = (md >= t["serious_months"]).astype(int)
    df[t["secondary"]] = (md >= 1).astype(int)
    return df


def prepare(df, cfg, with_targets=True):
    """Full pipeline: clean -> derived features -> targets."""
    out = add_derived(clean(df, cfg))
    if with_targets:
        out = add_targets(out, cfg)
    return out
