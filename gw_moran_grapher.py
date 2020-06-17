import random
import math
import numpy as np
import matplotlib.pyplot as plt

#################################################################
# Run a single Moran Sim
class Moran_Sim:
    def __init__(moran_sim, N, mutant_fitness):
        i = 1
        moran_sim.gen = 0
        while (True):
            i += moran_sim_step(i, mutant_fitness, N)
            moran_sim.gen += 1
            if (i == 0):
                moran_sim.success = False
                break
            if (i == N):
                moran_sim.success = True
                break
            
#################################################################   
# Perform one step of Moran Sim.
def moran_sim_step(i, r, N):
    val = random.random()
    dec_prob = float((N - i) * i) / ((r * i + N - i) * N)
    inc_prob = r * dec_prob
    if (val < dec_prob):
        return -1
    if (val > 1 - inc_prob):
        return 1
    else:
        return 0
    
#################################################################
# perform n_trials of Moran Sims with constant N and r.
# save freq and stderr as i.v.
class P_Trials:
    def __init__(p_trial, N, mutant_fitness, n_trials):
        successes = 0
        for k in range(n_trials):
            sim = Moran_Sim(N, mutant_fitness)
            if (sim.success):
                successes += 1
        p_trial.freq = float (successes) / n_trials
        p_trial.stderr = math.sqrt(p_trial.freq * (1 - p_trial.freq) / n_trials)

#################################################################
# perform Moran Sims with constant N and R until min_successes reached.
# save ave_time and stderr as i.v.
class T_Trials:
    def __init__(t_trial, N, mutant_fitness, min_successes):
        successes = 0
        total_time = 0
        total_sq_time = 0
        while(successes < min_successes):
            sim = Moran_Sim(N, mutant_fitness)
            if (sim.success):
                successes += 1
                total_time += sim.gen
                total_sq_time += sim.gen * sim.gen
        t_trial.ave_time = float (total_time) /min_successes
        var = float (total_sq_time) / (min_successes) - t_trial.ave_time * t_trial.ave_time
        t_trial.stderr = math.sqrt(var / (min_successes - 1))

#################################################################
# calculate P_fix(r, N)
def theoretical_prob(r, N): 
    return  (1 - 1/r)/(1 - 1/r**N)

#################################################################
# calculate E(fix_time(r, N)) for plots with varying r:
def theoretical_ave_time(r, N):
    # precompute all r**k values
    r_vals = np.empty(N+1)
    r_vals[0] = 1
    for k in range(1, N+1):
        r_vals[k] = r_vals[k-1] * r

    # compute values using precomputed values
    total = 0
    for k in range(1, N):
        for l in range (1, k+1):
            summand = 1/r_vals[k-l] - 1/r_vals[k]
            summand *= ((r * l + N - l) * N) / (r * l * (N - l))
            total += summand
    return total / (1 - 1/r_vals[N])

#################################################################
# Produce Graphs
def grapher(n_trials, lspace_val, fixed_N, min_r, inc_r, max_r,
            fixed_r, min_N, inc_N, max_N):
    rows, cols = 2, 2
    fig, axs = plt.subplots(rows, cols, sharex = 'col', sharey = 'row')

     # Plot p on r in top left
    ax = axs[0, 0]
    fitness_val = np.arange(min_r, max_r + inc_r, inc_r)
    y_val = np.empty(len(fitness_val))
    y_err = np.empty(len(fitness_val))
    # Calculate values
    for k in range(len(fitness_val)):
        sim = P_Trials(fixed_N, fitness_val[k], n_trials)
        y_val[k] = sim.freq
        y_err[k] = 2 * sim.stderr
    ax.errorbar(fitness_val, y_val, yerr = y_err, fmt = 'bx')
    # Plot theoretical values
     # Hard_coded to avoid div_0 errors
    x = np.linspace(min_r + inc_r , max_r + inc_r, lspace_val)
    y =  theoretical_prob(r = x, N = fixed_N)
    ax.plot(x, y)
    ax.set_ylim([0, 1])
    ax.set(ylabel = "Fixation Probability")

    # Plot t on r in bottom left
    ax = axs[1, 0]
    for k in range(len(fitness_val)):
        sim = T_Trials(fixed_N, fitness_val[k], n_trials)
        y_val[k] = sim.ave_time
        y_err[k] = 2* sim.stderr
    ax.errorbar(fitness_val, y_val, yerr = y_err, fmt = 'bx')
    # Plot theoretical values
    for k in range(len(x)):
        y[k] = theoretical_ave_time(r = x[k], N = fixed_N)
    ax.plot(x, y)
    ax.set_xlim([min_r - inc_r, max_r + inc_r])
    ax.set_yscale('log')
    ax.set_ylim([1, 10**4])
    ax.set(xlabel = "Selective Advantage of Mutant", ylabel = "Fixation Time")

    # Plot p on N in top right
    ax = axs[ 0, 1]
    pop_values = np.arange(min_N, max_N+ inc_N, inc_N)
    y_val = np.empty(len(pop_values))
    y_err = np.empty(len(pop_values))
    # Calculate values
    for k in range(len(pop_values)):
        sim = P_Trials(pop_values[k], fixed_r, n_trials)
        y_val [k] = sim.freq
        y_err[k] = 2 * sim.stderr
    ax.errorbar(pop_values, y_val, yerr = y_err, fmt = 'bx')
    # Plot theoretical values
    x = np.arange(min_N- inc_N+1, max_N + inc_N, 1)
    y =  theoretical_prob(r = fixed_r, N = x)
    ax.plot(x, y)
   

    # Plot t on N in bottom right
    ax = axs[1, 1]
    for k in range(len(pop_values)):
        sim = T_Trials(pop_values[k], fixed_r, n_trials)
        y_val [k] = sim.ave_time
        y_err[k] = 2 * sim.stderr
    ax.errorbar(pop_values, y_val, yerr =  y_err, fmt = 'bx')
    ax.set_xlim([min_N - inc_N, max_N + inc_N])
    ax.set(xlabel = "Population size")
    # Plot theoretical values
    for k in range(len(x)):
        y[k] = theoretical_ave_time(r = fixed_r, N = x[k])
    ax.plot(x, y)
    
    # Finish and save plot
    plt.savefig("Moran_sim.pdf")
    plt.show()

#################################################################
# MAIN METHOD
grapher(n_trials = 250, lspace_val = 100,  fixed_N = 50,  min_r = 1, inc_r = 0.2, max_r = 5,
        fixed_r = 1.1, min_N = 5, inc_N = 5, max_N = 100)

        


        
        
    
    
