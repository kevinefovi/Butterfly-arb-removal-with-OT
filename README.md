# OT Pre-processing of Maturity Slices

This repo finds and fixes calendar arbitrage in option term structures at some fixed log moneyness. It checks total variance across maturities, flagging any negative forward variance. A simple 1-D Optimal Transport right shifts the mass causing calendar spread arbitrage into a monotone curve, and clean IV is rebuilt. This is intended to act as a light pre processing layer before fitting a surface (SSVI) to make fits less rigid, especially at the wings.

The options data was a sample from Cboe Global Markets
