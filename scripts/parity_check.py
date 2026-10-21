"""Compare data.py (Python) cleaning with the dbt STAGING table. They must agree.

Run after `make dbt-run`:  python scripts/parity_check.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import snowflake.connector
import yaml
from dotenv import load_dotenv

from credit_risk.data import load_raw, prepare

load_dotenv()
root = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(root / "config" / "config.yaml"))

py = prepare(load_raw(root / cfg["paths"]["raw"]), cfg)
py_stats = {
    "rows": len(py),
    "serious_delinquency_sum": int(py["serious_delinquency"].sum()),
    "ever_delinquent_sum": int(py["ever_delinquent"].sum()),
    "cs_unknown_sum": int(py["cs_unknown"].sum()),
    "dti_unknown_sum": int(py["dti_unknown"].sum()),
    "ltv_gt80_sum": int(py["ltv_gt80"].sum()),
    "credit_score_mean": round(float(py["CreditScore"].mean()), 4),
    "dti_mean": round(float(py["DTI"].mean()), 4),
    "ltv_mean": round(float(pd.to_numeric(py["LTV"]).mean()), 4),
    "occupancy_unknown": int((py["Occupancy"] == "UNKNOWN").sum()),
    "channel_unknown": int((py["Channel"] == "UNKNOWN").sum()),
    "msa_unknown": int((py["MSA"] == "UNKNOWN").sum()),
    "num_borrowers_unknown": int((py["NumBorrowers"] == "UNKNOWN").sum()),
}

con = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"], role=os.environ["SNOWFLAKE_ROLE"],
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"], database=os.environ["SNOWFLAKE_DATABASE"],
    schema="STAGING",
)
cur = con.cursor()
cur.execute("""
    select count(*), sum(serious_delinquency), sum(ever_delinquent),
           sum(cs_unknown::int), sum(dti_unknown::int), sum(ltv_gt80::int),
           round(avg(credit_score), 4), round(avg(dti), 4), round(avg(ltv), 4),
           count_if(occupancy = 'UNKNOWN'), count_if(channel = 'UNKNOWN'),
           count_if(msa = 'UNKNOWN'), count_if(num_borrowers = 'UNKNOWN')
    from stg_loans
""")
row = [float(x) if x is not None else None for x in cur.fetchone()]
con.close()
sf_stats = dict(zip(py_stats.keys(), row))

bad = 0
print(f"{'check':26} {'python':>14} {'snowflake':>14}  result")
for k, v in py_stats.items():
    s = sf_stats[k]
    ok = s is not None and abs(float(v) - s) < 1e-3
    bad += not ok
    print(f"{k:26} {v:>14} {s:>14}  {'OK' if ok else 'MISMATCH'}")
print("\nPARITY OK" if bad == 0 else f"\n{bad} MISMATCH(ES): fix stg_loans.sql or data.py")
sys.exit(1 if bad else 0)
