# MOPSI

This project consists on calculating the gain of an american put option in the Black-Scholes model,
we start by a naive approach that takes a long time when the basket contains more than one asset 
(multidimensional amrican option), then we use the Longstaff-Schwarz algorith in order to estimate the expectation 
of gain by a monte carlo approach, and finally we use neuronal network in order to polish this estimation
