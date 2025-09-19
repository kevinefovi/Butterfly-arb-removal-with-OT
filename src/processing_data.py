import numpy as np
import pandas as pd
from pathlib import Path

path = Path("data/options_data.csv")
df = pd.read_csv(path)

df["quote_date"] = pd.to_datetime(df["quote_date"])
df["expiration"] = pd.to_datetime(df["expiration"])
df["dte_days"] = (df["expiration"] - df["quote_date"]).dt.days

df_spy = df[df["underlying_symbol"] == "SPY"].copy()

# defining anchors (buckets) to group expiries
anchors = [
    ("7D", 7, 2),
    ("14D", 14, 2),
    ("1M", 30, 3),
    ("2M", 60, 7),
    ("3M", 90, 10),
    ("6M", 180, 30),
    ("1Y", 365, 35)
]
anchor_df = pd.DataFrame(anchors, columns=["buckets", "anchor_days", "tolerance"])

# map each expiry to some anchor if it satisfies tolerance
def map_bucket(dte):
    best = None; best_err = np.inf 
    for name, a, tol in anchors:
        err = abs(dte - a)
        # keep memory of smallest distance to anchor
        if err <= tol and err < best_err:
            best = name; best_err = err
    return best

df_spy["buckets"] = df_spy["dte_days"].apply(map_bucket)

# pick one expiry per bucket (anchor)
# by distance to anchor, then liquidity

calls = df_spy[df_spy["option_type"] == "C"].copy()
calls["mid"] = (calls["bid_1545"] + calls["ask_1545"]) / 2.0
calls["spr"] = (calls["ask_1545"] - calls["bid_1545"]).clip(lower=0)
calls["rel_spr"] = np.where(calls["mid"] > 0, calls["spr"]/calls["mid"], np.nan)

# aggregate relative spread into median for each dte under each bucket
liq = (calls.groupby(["quote_date","expiration","buckets","dte_days"])
       .agg(rel_spread_med=("rel_spr","median"))
       .reset_index())

liq = pd.merge(liq, anchor_df, on="buckets", how="left")
liq["abs_err"] = (liq["dte_days"] - liq["anchor_days"]).abs()

# filter for df_spy rows that satisfy our ranking of dtes'
keep = (liq.sort_values(["quote_date","buckets","abs_err","rel_spread_med"])
        .drop_duplicates(["quote_date","buckets"])
        [["quote_date","expiration","buckets"]])

df_kept = df_spy.merge(keep, on=["quote_date","expiration","buckets"], how="inner")