import numpy as np
def analytic_solution(
    n = 50, initial_mut = 1, r_low = 1.1, r_high = 5.1
    ):
    #Create the fitness range
    r_range = np.arange(r_low, r_high, 0.05)
    output = []
    for r in r_range:
        fixation_prob = (1 - (1/r**initial_mut))/(1- (1/r**n))
        output.append([n, initial_mut, r, fixation_prob])
    return output