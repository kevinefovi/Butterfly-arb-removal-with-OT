import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, FuncFormatter

def fit_slice(y, m, strikes, call_targets, S0, R=1.0, tol=1e-10, max_iter=50):
    # future underlying levels at some fixed maturity
    y = np.asarray(y, float)
    # the baseline weights assigned to each y (our prior pmf)
    m = np.asarray(m, float)
    m = m / m.sum()
    K = np.asarray(strikes, float)
    # the moment targets our fitted law must match (fall inside the market bands)
    # + risk neutral mean (we assume R is 1 for simplicity reasons)) 
    targets = list(call_targets) + [S0*R, 1.0]

    # we a build a feature matrix where:
    # - rows are future underlying levels (states) 
    # - columns are functions evaluated at that state
    # so the matrix entries are payoffs evaluated at that state

    # the feature matrix's functions hold the function whose expectations will be matched
    payoffs = np.maximum(y[:, None] - K[None, :], 0.0)
    Phi = np.column_stack([payoffs, y[:, None], np.ones_like(y)[:, None]])
    targets = np.asarray(targets, float)
    nfeat = Phi.shape[1]

    # dual variables which when adjusted, titls the prior to hit the targets
    lam = np.zeros(nfeat)

    # generates a candidate probability distribution 
    # if you price calls under this probability distribution, the call price at K 
    # is the expected payoff of (S - K), so the call curve is butterfly free (decreasing in strike and convex)
    def pmf_from_lambda(lam):
        # keep the prior strictly positive (extremely small value to ensure well defined computations)
        eps = 1e-300
        m_pos = np.clip(m, eps, None)
        m_pos = m_pos / m_pos.sum()

        # we work in the log space
        # previously, w = m * exp(Phi @ lam) blew up when 'Phi @ lam' took on large values 
        # and if the prior has zeros 
        a = Phi @ lam
        logw = np.log(m_pos) + a

        # log sum exp normalization, preventing overflow
        c = np.max(logw)
        w = np.exp(logw - c)
        Z = np.sum(w)

        P = w / Z
        # return normalized P
        return P, (c + np.log(Z))


    for it in range(max_iter):
        P, Z = pmf_from_lambda(lam)
        moments = Phi.T @ P
        # mismatch between model and market
        g = moments - targets

        # stop if all moments match
        if np.linalg.norm(g, ord=np.inf) < tol:
            break

        # covariance of features under candidate probability distribution 
        E_phi_phiT = (Phi * P[:, None]).T @ Phi
        H = E_phi_phiT - np.outer(moments, moments)

        # works out what the new dual should look like
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Hessian H is a covariance matrix of features, so if some features are slightly linear 
            # cominbations of others, H has very small eigenvalues

            # So solving for H delta = g requires dividing by these tiny eigenvalues, blowing up delta 
            # we use gentle tikhonov and shift every eigenvalue up by some tiny e, so we get an invertile and well conditioned 
            # matrix
            H_reg = H + 1e-10 * np.eye(nfeat)
            delta = np.linalg.solve(H_reg, g)

        # full steps can be too aggressive, so we shrink the step size until the new iterate sees improvement
        step = 1.0
        for _ in range(20):
            P_try, _ = pmf_from_lambda(lam - step * delta)
            # if the new moment errors dont sufficiently decrease, we scale down step
            moments_try = Phi.T @ P_try
            if np.linalg.norm(moments_try - targets, np.inf) <= (1 - 0.5*step) * np.linalg.norm(g, np.inf):
                break
            step *= 0.5

        lam -= step * delta

    # final 
    P, _ = pmf_from_lambda(lam)
    model_calls = (payoffs.T @ P)
    return P, model_calls

def parity_regression(df, act_symbol, expiration):
    # OT solver assumes undiscounted targets
    df_slice = df[(df["act_symbol"]==act_symbol) & (df["expiration"]==expiration)].copy()
    df_slice["mid"] = 0.5*(df["bid"] + df["ask"])
    df_pivot = (df_slice.pivot_table(index=["act_symbol", "expiration", "strike"], columns="call_put", values="mid")
                .rename(columns={"Call":"Call_mid", "Put":"Put_mid"})
                .rename_axis(None, axis=1)
                .reset_index())
    
    # y = C - P, DF := intercept and D := slope on -K
    # design matrix X = [1, -K]
    K = df_pivot["strike"].values.astype(float)
    y = (df_pivot["Call_mid"] - df_pivot["Put_mid"]).values.astype(float)
    X = np.column_stack([np.ones_like(K), -K])
    
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = beta 

    # discount factor has to be positive, otherwise arbitrage 
    D = max(1e-12, slope)
    F = intercept / D

    # finally return undisc call targets, and we'll work in forward units
    df_pivot["undiscounted_call_targets"] = df_pivot["Call_mid"] / D
    strikes = df_pivot["strike"].values.astype(float)
    call_targets = df_pivot["undiscounted_call_targets"].values.astype(float)

    return strikes, call_targets, F, D

def plot_reweighting(y, m, P, title="Prior vs Fitted PMF (mass reweighting)"):

    idx = np.arange(len(y))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    b1 = ax.bar(idx - width/2, m, width, label="Prior pmf m", alpha=0.7, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(idx + width/2, P, width, label="Fitted pmf P", alpha=0.7, edgecolor="black", linewidth=0.5)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune=None))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xlabel("Future underlying level (grid)")

    ax.set_ylabel("Probability mass")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    plt.title(title)
    plt.tight_layout()
    plt.show()

df = pd.read_csv("data/raw_options.csv")
# we take the slice at "2019-02-15" for underlying "AAP" as an example
strikes, call_targets, forward, discount = parity_regression(df, "AAP", "2019-02-15")

# assume R = 1.0 and S0 := F
# most mass is within +-(k*vol*sqrt(T)), so we construct a grid on k=4
# and use a uniform prior for showcase

y = np.linspace(forward*0.75, forward*1.25, 200)
m = np.ones_like(y, dtype=float)
m /= m.sum()    

P, model_calls = fit_slice(y, m, strikes, call_targets, forward)
print(f"Undiscounted: {model_calls}\nDiscounted: {model_calls*discount}")

plot_reweighting(y, m, P)