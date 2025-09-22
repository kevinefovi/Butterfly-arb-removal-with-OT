# Entropic OT calibration of a single options slice 

I coded an aspect of 'Building arbitrage-free implied volatility: Sinkhorn’s algorithm and variants' by Hadrien De March and Pierre Henry-Labordère which, given vanilla quotes at 1 expiry and some pre defined prior, a risk neutral pmf P is found which reproduces the market's quotes and is butterfly arbitrage free by construction. The entropic OT solver works in undiscounted terms, so we estimated the discount factor and forward price via parity regression on a given slice. This repo is not an implementation of the paper, since we only fit 1 slice at a time meaning calendar arbitrage is not eliminated. 

Given our uniform prior, the implied call payoffs don't match the market targets. Over infinitely many exponential tilts of the prior, amongst the ones that reproduce the market's quotes and is butterfly arbitrage free, the KL closest one to the prior is picked.

#

The prior pmf and the fitted pmf are plotted, showing the movement of probability mass under a uniform prior for an example slice.

<img width="800" height="450" alt="final_fig" src="https://github.com/user-attachments/assets/d1c7dec1-9845-414f-9b9f-9ab73a7f3fcf" />

Example options data: https://www.dolthub.com/repositories/post-no-preference/options/data/master/option_chain
