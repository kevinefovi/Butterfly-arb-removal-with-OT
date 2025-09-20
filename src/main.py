import numpy as np
import pandas as pd 

"""
Given an EOD options surface:

- add_bucket_labels() bundles options into buckets
- pick_representative_expiries() picks exactly 1 expiry date per bucket (ranked by distance to anchor and then relative spread)

This processes the data which can be passed through:

- check_calendar_monotonicity() which detects calendar arbitrage at a given log moneyness slice (uses slice_through_maturity as helper)
- pass slice with calendar arbitrage to ot_processing which will update the IV to eliminate the negative forward variance 

Example anchors argument:

anchors = [
    ("7D", 7, 2),
    ("14D", 14, 2),
    ("1M", 30, 3),
    ("2M", 60, 7),
    ("3M", 90, 10),
    ("6M", 180, 30),
    ("1Y", 365, 35)
]
"""


def add_bucket_labels(df, 
                      anchors, 
                      buckets="buckets",
                      underlying_symbol="underlying_symbol",
                      symbol="SPY"):
    # labels options into buckets
    # assuming df has "quote_date" and "expiration"
    res = df[df[underlying_symbol]==symbol].copy()
    res["quote_date"] = pd.to_datetime(res["quote_date"])
    res["expiration"] = pd.to_datetime(res["expiration"])
    res["dte"] = (res["expiration"] - res["quote_date"]).dt.days

    def map_bucket(dte):
        best = None; best_err = np.inf
        for name, a, tol in anchors:
            err = abs(dte - a)
            if err <= tol and err < best_err:
                best, best_err = name, err
        return best

    res[buckets] = res["dte"].apply(map_bucket)
    return res

def pick_representative_expiries(df,
                                 anchors,
                                 option="C",
                                 option_type="option_type",
                                 quote="quote_date",
                                 expiration="expiration",
                                 buckets="buckets",
                                 bid="bid_1545",
                                 ask="ask_1545"):
    # from df buckets and dte columns, pick at most 1 expiry per bucket
    # ranked by distance to anchor then by liquidity 
    # returns a df with only the chosen expiries 

    anchor_df = pd.DataFrame(anchors, columns=["buckets", "anchor_days", "tolerance"])

    res = df.copy()
    res = res[res[option_type] == option]

    # mid / relative spread 
    res["mid"] = (res[bid] + res[ask]) / 2.0
    res["spr"] = (res[ask] - res[bid]).clip(lower=0)
    res["rel_spr"] = np.where(res["mid"] > 0, res["spr"] / res["mid"], np.nan)

    # median rel spread per expiry within bucket/day
    liq = (res.groupby([quote, expiration, buckets, "dte"])
             .agg(rel_spread_med=("rel_spr", "median"))
             .reset_index())

    liq = liq.merge(anchor_df, on="buckets", how="left")
    liq["abs_err"] = (liq["dte"] - liq["anchor_days"]).abs()

    # rank: closest to anchor, then tighter spread
    keep = (liq.sort_values([quote, buckets, "abs_err", "rel_spread_med"])
              .drop_duplicates([quote, buckets])[[quote, expiration, buckets]])

    # keep all rows (calls+puts if you want) for the chosen (date, expiry, bucket)
    df_kept = df.merge(keep, on=[quote, expiration, buckets], how="inner")
    return df_kept

# Calendar arbitrage logic 

def slice_through_maturities(df, 
                             m_target, 
                             strike="strike",
                             spot="active_underlying_price_1545"):

    # returns one option per bucket closest to the given log moneyness
    res = df.copy()
    res["m"] = np.log(res[strike] / res[spot])
    closest = (
        res.assign(dist=(res["m"] - m_target).abs())
        .sort_values(["buckets", "dist"])
        .drop_duplicates(subset="buckets", keep="first")
        .sort_values("dte")
        .reset_index(drop=True)
    )

    return closest

def check_calendar_monotonicity(df, 
                                m_target,
                                strike="strike",
                                spot="active_underlying_price_1545",
                                iv="implied_volatility_1545"):
    res = slice_through_maturities(df, m_target).copy()
    res["T"] = res["dte"] / 365.0

    # total variance (accumlating variance to T)
    res["w"] = (res["implied_volatility_1545"]**2)*res["T"]
    res["dt"] = res["T"].diff()
    res["dw"] = res["w"].diff()

    # forward variance
    res["fwd_var"] = res["dw"] / res["dt"] # slope of w 
    res["has_violation"] = res["fwd_var"] < 0
    has_arb = res["has_violation"].fillna(False).any()

    print("Calendar spread: ", "yes" if has_arb else "no")

def ot_processing(df, iv="implied_volatility_1545"):
    res = df.copy()
    res["T"] = res["dte"] / 365.0
    T = res["T"].to_numpy()
    iv = res[iv].to_numpy()
    w = (iv**2)*T

    dT = np.diff(T)
    dw = np.diff(w)
    fwd_var = dw / dT

    # 

    # pass the negative forward variance to the right
    fwd_var_star = fwd_var.copy()
    deficit = 0.0
    for i in range(len(fwd_var_star)):
        fwd_var_star[i] += deficit
        if fwd_var_star[i] < 0:
            deficit = fwd_var_star[i]
            fwd_var_star[i] = 0.0
        else:
            deficit = 0.0

    w_star = np.empty_like(w)
    w_star[0] = max(w[0], 0.0)
    for i in range(1, len(w)):
        w_star[i] = w_star[i-1] + fwd_var_star[i-1] * dT[i-1]
    iv_star = np.sqrt(np.maximum(w_star / T, 0.0))

    fwd_var_star = np.r_[np.nan, fwd_var_star]

    res["w_star"] = w_star
    res["iv_star"] = iv_star
    res["fwd_var_star"] = fwd_var_star

    return res