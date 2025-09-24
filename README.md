# Kullback-Leibler projection of prior onto options slice 

Given vanilla quotes at 1 expiry and a strictly positive prior, a risk neutral pmf P is found such that for each strike, the undiscounted model call value matches the market target. Because call prices are expectations under a probability measure, the calibrated slice is automatically decreasing and convex in strike (free of butterfly arbitrage). This calibrates 1 slice at a time, so calendar arbitrage isn't removed in this repo. The solver works in undiscounted terms, so we estimated the discount factor and forward via a put call parity regression on a given slice.

Starting from a simple uniform prior, the implied call payoffs initially don't match the market. We solve in the dual, optimizing over the Lagrange multipliers lambda associated with the undiscounted call payoff moments, the mean and total mass. This yields the exponential tilt which is the KL closest distribution to the prior that satisfies the constraints. If new moment errors don't sufficiently decrease, we backtrack by halving the step size, skipping unproductive regions of the search. Convergence is relatively quick because of this.

Most importantly, priors allow you to encode information about the market into your model and the KL projection step then moves the minimal amount of 'mass' to satisfy the constraints, leaving your structure (beliefs) intact elsewhere.

#

The prior pmf and the fitted pmf are plotted for an example slice, showing the movement of probability mass under a uniform prior.

<img width="800" height="450" alt="Figure_2" src="https://github.com/user-attachments/assets/6bf1ceec-e954-46bc-b71b-b709282bba3c" />

Symbol: "AAP", expiry: "2019-02-15"

Example options data: https://www.dolthub.com/repositories/post-no-preference/options/data/master/option_chain

# 

The obvious next step is to extend to multiple maturities and impose cross maturity constraints to achieve a static arbitrage free surface overall.
