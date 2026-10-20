"""Load the real Freddie Mac CSV into Snowflake RAW.LOANS_REAL, untouched (all text).

Safe to rerun: the table is replaced each time, so rows are never duplicated.
Usage:  python scripts/load_raw.py --csv data/raw/FreddieMac.csv
"""
import argparse
import datetime as dt
import os
import re
from pathlib import Path

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector.pandas_tools import write_pandas


def connect(schema="RAW"):
    load_dotenv()
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=schema,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/FreddieMac.csv")
    ap.add_argument("--table", default="LOANS_REAL")
    args = ap.parse_args()

    # dtype=str and keep_default_na=False keep every value exactly as in the file
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    df.columns = [re.sub(r"\W+", "_", c.strip()).upper() for c in df.columns]
    df["_SOURCE_FILE"] = Path(args.csv).name
    df["_LOADED_AT"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    print(f"Read {len(df):,} rows, {df.shape[1]} columns from {args.csv}")

    con = connect()
    try:
        ok, n_chunks, n_rows, _ = write_pandas(
            con, df, args.table.upper(),
            database=os.environ["SNOWFLAKE_DATABASE"], schema="RAW",
            auto_create_table=True, overwrite=True, quote_identifiers=False,
        )
        cur = con.cursor()
        cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT LOANSEQNUM) FROM RAW.{args.table.upper()}")
        total, distinct = cur.fetchone()
        print(f"write ok={ok}, rows written={n_rows:,}")
        print(f"In Snowflake: {total:,} rows, {distinct:,} distinct LOANSEQNUM")
        assert total == len(df), "row count mismatch between file and table"
    finally:
        con.close()


if __name__ == "__main__":
    main()
    