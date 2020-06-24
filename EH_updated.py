#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:40:48 2020

@author: Emily
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:05:03 2020

@author: Emily
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def moran_step(nA, r, N):
    
    flip = np.random.rand()
    less = (N - nA)/(r * nA + N - nA) * nA/N
    more = r*less
    
    if flip < less:
        return -1
    if flip > (1-more):
        return 1
    else:
        return 0

#########################################################
# part 1: simulate 50 replicates each of N=1000, starting 50% for a range of r
pop_size = 1000
fit_vals = np.array([0.5, 0.75, 0.9, 0.99, 1, 1.01, 1.1, 1.5, 2]) 
replicates = 10
time = 5000
traces = np.empty([len(fit_vals),time,replicates])
traces[:,0,:] = pop_size/2
trace_ave = np.zeros([len(fit_vals),time])
trace_err = np.zeros([len(fit_vals),time])

for j,r in enumerate(fit_vals):
    
    for k in range(replicates):
        
        for t in range(1, time):
            
            if traces[j,t-1,k] ==0 or traces[j,t-1,k] == pop_size:    
                traces[j,t,k] = traces[j,t-1,k]
                
            else:
                traces[j,t,k] = traces[j,t-1,k] + moran_step(traces[j,t-1,k],r,pop_size)
    
    for t in range(time):
        trace_ave[j,t] = np.average(traces[j,t,:])
        trace_err[j,t] = np.std(traces[j,t,:])
                 
    # Plot
    plt.fill_between(np.arange(time), 
                      trace_ave[j,:] - trace_err[j,:],
                      trace_ave[j,:] + trace_err[j,:])
    plt.ylim(0,pop_size)
    plt.title('N=1000')
    plt.legend(['0.5', '0.75', '0.9', '0.99', '1', '1.01', '1.1', '1.5', '2'])
    plt.xlabel('Generation #')
    plt.ylabel('Proportion of population that is A');
    
#########################################################
# part 2: Extinction times for various r, Nsimulate 50 replicates each of N=1000, starting 50% for a range of r
pop_sizes = np.array([10,100,1000]) 
fit_vals = np.array([1.1, 1.4, 1.7, 2]) 
replicates = 10
fix_times = np.empty([len(pop_sizes),len(fit_vals),replicates])

for i,N in enumerate(pop_sizes):
    
    for j,r in enumerate(fit_vals):
        
        for k in range(replicates):
            
            trace =  N/2
            fix_times[i,j,k] = 0
            
            while trace > 0 and trace < N:
                
                trace += moran_step(trace, r, N)
                fix_times[i,j,k] += 1
    fix_times[i,:,:] /= N

# reformat data into a df
shape = fix_times.shape
index = pd.MultiIndex.from_product([range(s)for s in shape], names=['N', 'r','replicate'])
df = pd.DataFrame({'fix_times': fix_times.flatten()}, index=index).reset_index()

# pretty colors
sns.boxplot(x="r", y="fix_times", hue="N", data=df, palette="mako_r")
plt.xlabel("Fitness") 
plt.ylabel("Fixation Time / Pop Size")
plt.title("Fixation Time");
plt.legend(['10','100','1000'])
plt.xticks([0, 1, 2, 3], ['1.1', '1.4', '1.7', '2'])
    
#########################################################
# part 3: Extinction times for various r, N
        
replicates = 1000

# Fitness sweep

N = 50
fit_vals = np.linspace(1,5,21) 

fix_times = np.empty([len(fit_vals),replicates])
fix_time_ave = np.empty([len(fit_vals)])
fix_time_err = np.empty([len(fit_vals)])
fix_prob = np.ones([len(fit_vals)])
fix_prob_err = np.empty([len(fit_vals)])

for j,r in enumerate(fit_vals):
    
    for k in range(replicates):
        
        trace =  1
        fix_times[j,k] = 0

        while trace > 0 and trace < N:
            
            trace += moran_step(trace, r, N)
            fix_times[j,k] += 1
        
        if trace == 0: 
            
            fix_times[j,k] = np.nan #only counts if it fixates
            fix_prob[j] -= 1/replicates
             
    fix_time_ave[j] = np.nanmean(fix_times[j,:]) 
    fix_time_err[j] = np.nanstd(fix_times[j,:])
    fix_prob_err[j] = 2*np.sqrt(fix_prob[j]*(1-fix_prob[j])/replicates)
    

# Plots
fig = plt.figure()
fig.suptitle("Well-mixed Moran Process, 1000 simulations")
fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
# 3a: fixation probability vs fitness
theoretical = np.empty([len(fit_vals)])
for j,r in enumerate(fit_vals):
    theoretical[j] = (1-1/r)/(1-r**-N)
theoretical[0] = 1/N
plt.subplot(2,2,1)
plt.plot(fit_vals,theoretical)
plt.errorbar(fit_vals,fix_prob,fix_prob_err)
plt.ylabel("Fixation Probability")
plt.ylim(0,1)

# 3b: fixation time vs. fitness
plt.subplot(2,2,3)
plt.plot(fit_vals,fix_time_ave)
plt.errorbar(fit_vals, fix_time_ave, fix_time_err,)
plt.yscale("log")
plt.ylim(100,10000)
plt.xlabel("Fitness") 
plt.ylabel("Fixation Time")


# Population size sweep
r = 1.1
pop_vals = np.linspace(5,100,20)

fix_times = np.empty([len(pop_vals),replicates])
fix_time_ave = np.empty([len(pop_vals)])
fix_time_err = np.empty([len(pop_vals)])
fix_prob = np.ones([len(pop_vals)])
fix_prob_err = np.empty([len(pop_vals)])

for i,N in enumerate(pop_vals):
    
    for k in range(replicates):
        
        trace =  1
        fix_times[i,k] = 0

        while trace > 0 and trace < N:
            
            trace += moran_step(trace, r, N)
            fix_times[i,k] += 1
            
        if trace == 0: 
            
            fix_times[i,k] = np.nan #only counts if it fixates
            fix_prob[i] -= 1/replicates
        
    fix_time_ave[i] = np.nanmean(fix_times[i,:]) 
    fix_time_err[i] = np.nanstd(fix_times[i,:])
    fix_prob_err[i] = 2*np.sqrt(fix_prob[i]*(1-fix_prob[i])/replicates)

    
# 3c: fixation probability vs fitness
theoretical = np.empty([len(pop_vals)])
for i,N in enumerate(pop_vals):
    theoretical[i] = (1-1/r)/(1-r**-N)

plt.subplot(2,2,2)
plt.plot(pop_vals,theoretical)
plt.errorbar(pop_vals,fix_prob,fix_prob_err)
plt.ylim(0,1)

# 3d: fixation time vs. fitness
plt.subplot(2,2,4)
plt.plot(pop_vals,fix_time_ave)
plt.errorbar(pop_vals, fix_time_ave, fix_time_err)
plt.yscale("log")
plt.ylim(10,10000)
plt.xlabel("Population size") 








