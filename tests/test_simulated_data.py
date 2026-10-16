"""Integrity checks on generated simulation files. Skipped (returns) if you have not generated them yet."""
import glob
from pathlib import Path

import pandas as pd

SIM = Path(__file__).resolve().parents[1] / "data" / "sim"
OUTCOME_COLS = {"EverDelinquent", "MonthsDelinquent", "MonthsInRepayment"}


def _files(kind):
    return sorted(glob.glob(str(SIM / kind / "*.csv")))


def test_batches_contain_no_outcome_columns():
    for f in _files("batches")[:3]:
        assert not (OUTCOME_COLS & set(pd.read_csv(f, nrows=5).columns)), f


def test_outcomes_are_internally_consistent():
    for f in _files("outcomes")[::6]:
        o = pd.read_csv(f)
        assert ((o.EverDelinquent == 0) == (o.MonthsDelinquent == 0)).all(), f
        assert (o.MonthsDelinquent <= 24).all(), f
        assert o.LoanSeqNum.is_unique, f


def test_every_outcome_matches_a_loan():
    for fb, fo in list(zip(_files("batches"), _files("outcomes")))[::8]:
        loans = set(pd.read_csv(fb, usecols=["LoanSeqNum"]).LoanSeqNum)
        outc = set(pd.read_csv(fo).LoanSeqNum)
        assert outc <= loans and len(outc) > 0


def test_simulated_rows_are_labelled():
    for f in _files("batches")[::8]:
        b = pd.read_csv(f, usecols=["LoanSeqNum", "DataSource"])
        assert (b.DataSource == "SIMULATED").all()
        assert b.LoanSeqNum.str.startswith("S").all()
