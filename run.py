import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
from scipy.special import ndtr

# ---------- Black-76 (undiscounted) ----------
SQRT2PI = np.sqrt(2*np.pi)
def npdf(x): 
    return np.exp(-0.5*x*x)/SQRT2PI

def black_call_undisc(F,K,T,s):
    if T<=0 or s<=0: return max(F-K,0.0)
    vs=s*np.sqrt(T); d1=(np.log(F/K)+0.5*s*s*T)/vs; d2=d1-vs
    return F*ndtr(d1)-K*ndtr(d2)

def vega_undisc(F,K,T,s):
    if T<=0 or s<=0: return 0.0
    vs=s*np.sqrt(T); d1=(np.log(F/K)+0.5*s*s*T)/vs
    return F*npdf(d1)*np.sqrt(T)

def iv_from_call_undisc(F,K,T,Ct, lo=1e-6, hi=5.0):
    # Guard trivial bounds: 0 <= Ct <= F
    if Ct < max(F-K,0)-1e-12 or Ct>F+1e-12: return np.nan
    f = lambda s: black_call_undisc(F,K,T,s)-Ct
    # ensure bracket
    if f(lo)*f(hi)>0:
        hi2=10.0
        if f(lo)*(black_call_undisc(F,K,T,hi2)-Ct)>0: return np.nan
        return brentq(lambda s: black_call_undisc(F,K,T,s)-Ct, lo, hi2, maxiter=100)
    return brentq(f, lo, hi, maxiter=100)

# ---------- Parity regression: infer D, F per expiry ----------
def parity_forward_discount(g):

    gc = g[g["option_type"]=="C"].copy()
    gp = g[g["option_type"]=="P"].copy()

    gc["mid"]=(gc["bid_1545"] + gc["ask_1545"]) / 2
    gp["mid"]=(gp["bid_1545"] + gp["ask_1545"]) / 2

    m = pd.merge(gc[["strike","mid"]].rename(columns={"mid":"C"}),
                 gp[["strike","mid"]].rename(columns={"mid":"P"}), on="strike", how="inner").dropna()

    Y=(m["C"] - m["P"]).values
    X=m["strike"].values
    b,a = np.polyfit(X,Y,1)            # Y = a + b*K
    D = -b                              # slope = -D
    F = a/(-b)       # intercept = D*F
    
    return pd.Series({"D":D, "F":F})

# ---------- Feature builder for calls ----------
def build_call_features(df):
    # requires df already filtered to chosen expiries (one per bucket per date)
    df = df.copy()
    df["quote_date"]=pd.to_datetime(df["quote_date"])
    df["expiration"]=pd.to_datetime(df["expiration"])
    df["T"] = (df["expiration"]-df["quote_date"]).dt.days/365.0

    # Parity per (date, expiry)
    FD = (df.groupby(["quote_date","expiration"], group_keys=False).apply(parity_forward_discount, include_groups=False)
            .reset_index())
    df = df.merge(FD, on=["quote_date","expiration"], how="left")

    # Calls only
    c = df[df["option_type"].str.upper().str[0]=="C"].copy()
    c["mid"] = (c["bid_1545"]+c["ask_1545"])/2
    c["spread"] = (c["ask_1545"]-c["bid_1545"]).clip(lower=0)
    c["Ctilde"] = c["mid"]/c["D"]

    # Invert IV, compute vega & k
    ivs=[]
    for F,K,T,Ct in zip(c["F"].values, c["strike"].values, c["T"].values, c["Ctilde"].values):
        ivs.append(iv_from_call_undisc(F,K,T,Ct))
    c["iv"]=ivs
    c = c.replace([np.inf,-np.inf], np.nan).dropna(subset=["iv"])
    c["vega"] = [vega_undisc(F,K,T,s) for F,K,T,s in zip(c["F"],c["strike"],c["T"],c["iv"])]
    c["k"] = np.log(c["strike"]/c["F"])

    # Baseline weights: vega / spread^2 with caps
    eps=1e-12
    w = c["vega"]**2 / (np.maximum(c["spread"], eps)**2)
    if (w>0).any():
        cap = np.quantile(w[w>0], 0.99)
        w = np.minimum(w, cap)
    c["w_base"] = w
    return c

df_spy = pd.read_csv("data/df_kept.csv", index_col=0)
df = build_call_features(df_spy)
df.to_csv("data/df_calls.csv")