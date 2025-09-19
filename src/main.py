import numpy as np
import pandas as pd
from pathlib import Path

pth = Path("data/df_kept.csv")
df = pd.read_csv(pth, index_col=0)
df["T"] = df["dte_days"] / 365.0

def slice_through_maturities(df, tgt):
    d = df.copy()
    d["m"] = np.log(d["strike"] / d["active_underlying_price_1545"])
    closest = (
        d.assign(dist=(d["m"] - tgt).abs())
        .sort_values(["buckets", "dist"])
        .drop_duplicates(subset="buckets", keep="first")
        .sort_values("dte_days")
        .reset_index(drop=True)
    )

    return closest

def check_calendar_monotonicity(df, m):
    res = slice_through_maturities(df, m).copy()
    # total variance (accumlating variance to T)
    res["w"] = (res["implied_volatility_1545"]**2)*res["T"]
    res["dt"] = res["T"].diff()
    res["dw"] = res["w"].diff()

    # forward variance
    res["fwd_var"] = res["dw"] / res["dt"] # slope of w 
    res["has_violation"] = res["fwd_var"] < 0
    has_arb = res["has_violation"].fillna(False).any()

    print("Calendar spread: ", "yes" if has_arb else "no")

    return res

# above functions were used to detect calendar arbitrage 

def ot_processing(df):
    d = df.copy()
    T = d["T"].to_numpy()
    iv = d["implied_volatility_1545"].to_numpy()
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

    d["w_star"] = w_star
    d["iv_star"] = iv_star
    d["fwd_var_star"] = fwd_var_star

    return d
