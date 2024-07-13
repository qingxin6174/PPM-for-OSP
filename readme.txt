
[aDG,aNE] = oppm(T,idx,delay) OPPM
T: Total rounds, idx: Case index, aDG: average duality gap, aNE: average NE-reg

[aDG,aNE] = optoppm(T,idx,delay) OptOPPM
T: Total rounds, idx: Case index, delay = 4, aDG: average duality gap, aNE: average NE-reg

[aDG,aNE] = hedgeoptoppm(T,idx,delay) OptOPPM with Multiple Predictors 
T: Total rounds, idx: Case index, delay = [4,5,6], aDG: average duality gap, aNE: average NE-reg

opt_entropy: Solver for Clipped Hedge

payoff: return the payoff function
