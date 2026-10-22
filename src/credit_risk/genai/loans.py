"""Look up a loan from the dataset by its loan ID.

The assistant never takes loan numbers from the LLM. When someone asks about a loan ID, this module reads the row
from the loan file, keeps ONLY the approved model inputs (no postal code, no seller, no servicer, no location),
and hands that record to the scorecard. Loans can still be typed in by hand when they are not in the dataset.
"""
import numpy as np
import pandas as pd

from .. import data
from ..config import load_config
from . import governance as gov

ID_COLUMN = "LoanSeqNum"


class LoanStore:
    """Loan file loaded on first use (a large CSV takes a few seconds), then kept in memory by loan ID."""

    def __init__(self, path=None, cfg=None):
        self.cfg = cfg or load_config()
        self.path = path or self.cfg["paths"]["raw"]
        self._df = None

    @property
    def df(self):
        if self._df is None:
            try:
                raw = data.load_raw(self.path)
            except FileNotFoundError:
                raise gov.GovernanceError(f"I can't find the loan file at {self.path}, so I can't look up loan IDs. "
                                          f"You can still type the loan details in.")
            raw = raw.drop_duplicates(ID_COLUMN)
            raw["_key"] = raw[ID_COLUMN].astype("string").str.upper()
            self._df = raw.set_index("_key", drop=False)
        return self._df

    def __contains__(self, loan_id):
        return str(loan_id).strip().upper() in self.df.index

    def find_ids(self, text):
        """Loan IDs from the file that appear in `text` (in order of appearance)."""
        import re
        found = []
        for tok in re.findall(r"[A-Za-z0-9]{8,14}", text):
            t = tok.upper()
            if any(c.isdigit() for c in t) and any(c.isalpha() for c in t) and t in self.df.index and t not in found:
                found.append(t)
        return found

    def get(self, loan_id):
        """Return (loan, info). `loan` has model inputs only; `info` carries the recorded outcome for context."""
        key = str(loan_id).strip().upper()
        if key not in self.df.index:
            raise gov.GovernanceError(f"I can't find loan {loan_id} in the loan file. Check the ID, or type the loan details in.")
        return self._row_to_loan(self.df.loc[key])

    def random(self, seed=None):
        """A random loan that has all the required model inputs."""
        rows = self.df
        ok = rows[(pd.to_numeric(rows["CreditScore"], errors="coerce") > 0) & (pd.to_numeric(rows["DTI"], errors="coerce") > 0)]
        row = (ok if len(ok) else rows).sample(1, random_state=seed).iloc[0]   # prefer loans with every input recorded
        return self._row_to_loan(row)

    def _row_to_loan(self, row):
        loan = {}
        for f in gov.NUMERIC_RANGES:
            v = pd.to_numeric(row.get(f), errors="coerce")
            if f in gov.UNKNOWN_IS_ZERO and (pd.isna(v) or v == 0):         # unknown stays 0; the scorecard has a flag for it
                loan[f] = 0.0
                continue
            if pd.notna(v):
                loan[f] = float(v)
        for f, allowed in gov.CATEGORY_VALUES.items():
            v = row.get(f)
            if v is None or pd.isna(v):
                continue
            s = str(v).strip().upper()
            if s.endswith(".0"):
                s = s[:-2]
            loan[f] = s if s in allowed else "X"
        missing = [k for k in ("LTV", "OrigUPB", "OrigInterestRate") if k not in loan]
        if missing:
            raise gov.GovernanceError(
                f"Loan {row[ID_COLUMN]} has no recorded {', '.join(gov.LABELS[m] for m in missing)}, so the scorecard "
                f"can't score it as is. You can type the missing value in.")
        clean = gov.validate_fields(loan)
        info = {"loan_id": str(row[ID_COLUMN])}
        md = pd.to_numeric(row.get("MonthsDelinquent"), errors="coerce")
        if pd.notna(md):
            info["months_delinquent"] = int(md)
        return clean, info
