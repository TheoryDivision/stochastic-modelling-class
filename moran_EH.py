#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:05:03 2020

@author: Emily
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from ggplot import *

def moran(pop_size, time):
    
    tracker = np.zeros([time,pop_size])
    trace = np.empty([time])
    test = 5

    for i in range(pop_size):
        if i < pop_size/2:
            tracker[0,i] = 1
    trace[0] = pop_size/2
            
    for t in range(1,time):
        tracker[t,:] = tracker[t-1,:]
        divide = int(np.floor(pop_size*np.random.rand()))
        die = int(np.floor(pop_size*np.random.rand()))
        tracker[t,die] = tracker[t,divide]
        trace[t] = np.sum(tracker[t,:])
    return trace, test

# trace, test = moran(100,100)

def weighted_moran(pop_size, time, r):
    
    trace = np.empty([time])
    trace[0] = pop_size/2 # initialize half and half
    
    fix_time = 0
        
    for t in range(1,time):
        
        trace[t] = trace[t-1]

        if trace[t] > 0 and trace[t] < pop_size:
            
            prob_A = r*trace[t-1]
            prob_B = pop_size-trace[t-1]
            N = prob_A + prob_B
            
            divide = N*np.random.rand()
            die = pop_size*np.random.rand()
            
            # A divides, B dies
            if divide < prob_A and die > trace[t-1]:
                trace[t] = trace[t-1] + 1 
                
            # B divides, A dies
            elif divide > prob_A and die < trace[t-1]:
                trace[t] = trace[t-1] - 1 
                
            fix_time += 1
            
        # else: print('no change at time', t)
                
    return trace, fix_time  


pop_sizes = np.array([1000])
fit_vals = np.array([1.1]) #np.linspace(1,5,21) # np.array([0.5, 0.75, 0.9, 0.99, 1, 1.01, 1.1, 1.5, 2]) #
replicates = 10
time = 5000
# time_sc = np.array([time,len(pop_sizes)])
traces = np.empty([len(pop_sizes),len(fit_vals),time,replicates])
trace_ave = np.zeros([len(pop_sizes),len(fit_vals),time])
trace_err = trace_ave
fix_times = np.empty([len(pop_sizes),len(fit_vals),replicates])
fix_prob = np.empty([len(pop_sizes),len(fit_vals)])
frean = np.empty([21])

for i,pop_size in enumerate(pop_sizes):
    
    for j,r in enumerate(fit_vals):
        
        for k in range(replicates):
            traces[i,j,:,k], fix_times[i,j,k] = weighted_moran(pop_size,time,r)
            # traces[i,j,:,k], fix_times[i,j,k] = weighted_moran(pop_size,time,r)/pop_size 
            # plt.plot(np.arange(time), traces[i,j,:,k])
            
        frean[j] = np.average(fix_times[i,j,:]) #only do this if one pop_size
        
        for t in range(time):
            
            trace_ave[i,j,t] = np.average(traces[i,j,t,:])
            # trace_err[i,j,t] = np.std(traces[i,j,t,:])
                     
        # Plot
        # plt.plot(np.arange(time), trace_ave[i,j,:], 'o')
        
        # fix_prob[i,j] = np.sum(fix_times[i,j,:]<time-1)/pop_size #only do this if one pop_size

# # frean
# plt.plot(fit_vals, fix_prob[0,:], 'o')
# # plt.yscale("log")
# plt.title('N = 50, 25 replicates')
# plt.xlabel('Relative fitness')
# plt.ylabel('Fixation probability')

# d = np.transpose(fix_times[0,:,:])
# df = pd.DataFrame(d, columns=['r=1.2', 'r=1.5', 'r=2'])

# data = pd.DataFrame({'X': fix_times[:], 'Y': Y})
# 0. Visualize data distribution
# data.boxplot(by='', meanline=True, showmeans=True)
# plt.show()
# df.boxplot()
# plt.ylabel('Fixation time/population')
# plt.title('Population 1000')

# # Couldn't get this to work
# sns.set(style="whitegrid")
# ax = sns.catplot(x=fit_vals,y=fix_times[0,:,:])
    
# data = [fix_times[0,2,:],fix_times[1,2,:],fix_times[2,2,:]]
# fig, ax = plt.subplots()
# # ax.set_title('Fixation times for N=200')
# ax.boxplot(data)    
# plt.show()

# plot
# plt.title('N=1000')
# plt.legend(['0.5', '0.75', '0.9', '0.99', '1', '1.01', '1.1', '1.5', '2'])
# plt.xlabel('Generation #')
# plt.ylabel('Proportion of population that is A');
 
# import numpy as np
# import matplotlib.pyplot as plt
# #generate data
# t=5000 # total time steps
# K_space = np.array([100,1000]) #pop sizes. to iterate over
# r_space = np.array([1.1,1.5,2]) #mut fit to iterate over
# samp = 20 # sample num
# results = np.zeros((len(K_space),len(r_space),samp,t)) # array to store results
# for i,K in enumerate(K_space):
#     for j, r in  enumerate(r_space):
#         for n in range(samp):
#             #4D array storing mutant B frequency over each time step, for each sample number, pop size and mut rate
#             results[i,j,n,:]=moran_process(t,K,f_0,r)[:,1]/K  #a single simulation over t time steps is run by function 'moran_process'
# #GRAPHING
# fig,ax= plt.subplots(figsize= (9,6))
# #colors to distuingsh r values
# col=['c','m','y']
# #alpha values to distinguish Ks
# a =[.4,.8]     
# #iterate back over array 4d results array
# #storing mutant B frequency over each time step, for each sample number, pop size and mut rate
# for i,K in enumerate(K_space):
#     for j, r in  enumerate(r_space):
#         # median of data over all samples for each condition
#         y = np.median(np.array(results[i,j]),axis=0)[:-1]
#         yerr =np.std(np.array(results[i,j]),axis=0)[:-1]
#         ax.plot(np.arange(t)[:-1],y,color = 'darkgrey')
#         # error env. from standard deviation
#         ax.fill_between(np.arange(t)[:-1], y-yerr, [min(ye,1) for ye in y+yerr],alpha=a[i],color = col[j],label='K= %d,r= %0.2f'% (K, r)) 
# ax.legend()
# ax.set_ylabel('B',fontsize=15)
# ax.set_xlabel('Time step',fontsize=15)
# ax.set_ylim([.4,1.1])
# for axis in ['bottom','left']:
#     ax.spines[axis].set_linewidth(2)
# for axis in ['top','right']:
#     ax.spines[axis].set_linewidth(0)
    

    
