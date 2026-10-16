"""Data cleaning, target and derived-feature tests, using the 100-row real-data fixture.

Fixture rows with deliberate edge cases (see scripts/simulate_batches.py::make_fixture):
  0 CreditScore=0 | 3 DTI=0 | 4 MonthsDelinquent=2 | 5 MonthsDelinquent=3 | 7 FirstTimeHomebuyer='X'
"""
from pathlib import Path

import pandas as pd

from credit_risk import data
from credit_risk.config import load_config

FIXTURE = Path(__file__).parent / "fixtures" / "loans_sample.csv"
CFG = load_config()


def _raw():
    return data.load_raw(FIXTURE)


def test_target_boundary_two_vs_three_months():
    d = data.prepare(_raw(), CFG)
    assert d.loc[4, "serious_delinquency"] == 0      # 2 months late: not serious
    assert d.loc[5, "serious_delinquency"] == 1      # 3 months late: serious
    assert d.loc[4, "ever_delinquent"] == 1 and d.loc[5, "ever_delinquent"] == 1
    assert d.loc[6, "serious_delinquency"] == 0 and d.loc[6, "ever_delinquent"] == 0


def test_serious_implies_ever():
    d = data.prepare(_raw(), CFG)
    assert (d["serious_delinquency"] <= d["ever_delinquent"]).all()


def test_unknown_x_code_becomes_unknown_category():
    d = data.prepare(_raw(), CFG)
    assert d.loc[7, "FirstTimeHomebuyer"] == "UNKNOWN"
    for c in CFG["features"]["categorical"]:
        assert not d[c].isin(["X", ""]).any(), c


def test_zero_sentinels_become_missing_with_flag():
    d = data.prepare(_raw(), CFG)
    assert pd.isna(d.loc[0, "CreditScore"]) and d.loc[0, "cs_unknown"] == 1
    assert pd.isna(d.loc[3, "DTI"]) and d.loc[3, "dti_unknown"] == 1
    assert d.loc[1, "cs_unknown"] == 0 or pd.isna(d.loc[1, "CreditScore"])


def test_clean_does_not_modify_input():
    raw = _raw()
    before = raw.copy()
    data.clean(raw, CFG)
    pd.testing.assert_frame_equal(raw, before)


def test_text_columns_are_stripped():
    raw = _raw()
    for c in raw.columns:
        if not pd.api.types.is_numeric_dtype(raw[c]):
            s = raw[c].dropna().astype(str)
            assert (s == s.str.strip()).all(), c


def test_vintage_quarter():
    df = pd.DataFrame({"FirstPaymentDate": [199903, 200108, 200012], "LTV": [80, 81, 10]})
    out = data.add_derived(df)
    assert list(out["vintage_quarter"]) == ["1999Q1", "2001Q3", "2000Q4"]


def test_ltv_gt80_flag_boundary():
    df = pd.DataFrame({"LTV": [79, 80, 81]})
    assert list(data.add_derived(df)["ltv_gt80"]) == [0, 0, 1]


def test_model_inputs_exclude_outcomes_and_proxies():
    from credit_risk.config import model_inputs
    inputs = set(model_inputs(CFG))
    for bad in ("MonthsDelinquent", "MonthsInRepayment", "EverDelinquent", "ServicerName",
                "PropertyState", "MSA", "SellerName", "PostalCode"):
        assert bad not in inputs, bad
