# Entropic OT calibration of a single options slice 

Given vanilla quotes at 1 expiry and a positive prior, a risk neutral pmf P is found such that for each strike, the undiscounted model call value matches the market target. Since prices are expectations under a pmf, the slice is free of butterfly arbitrage. This doesn't completely eliminate static arbitrage as only 1 slice at a time is calibrated, meaning calendar arbitrage isn't constrained for. The entropic OT solver works in undiscounted terms, so we estimated the discount factor and forward price via parity regression on a given slice.

Given the naive selection of a uniform prior, the implied call payoffs don't match the market. The procedure solves in the dual (undiscounted call payoffs, mean and mass) instead of optimizing over the entire pmf P on an m point grid (high dimensional). Convergence is relatively quick because of this.

Most importantly, priors allow you to encode information about the market into your model and the entropic OT step then moves the minimal amount of 'mass' to satisfy the constraints, leaving your structure (beliefs) intact elsewhere.

#

The prior pmf and the fitted pmf are plotted for an example slice, showing the movement of probability mass under a uniform prior.

<img width="800" height="450" alt="Figure_2" src="https://github.com/user-attachments/assets/6bf1ceec-e954-46bc-b71b-b709282bba3c" />

Symbol: "AAP", expiry: "2019-02-15"

Example options data: https://www.dolthub.com/repositories/post-no-preference/options/data/master/option_chain

# 

The obvious next step is to extend to multiple maturities to remove static arbitrage all in all.
