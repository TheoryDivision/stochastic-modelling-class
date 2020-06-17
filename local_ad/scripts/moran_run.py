#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import FormatStrFormatter

def l_norm(x=None):
    return x/np.sum(x)

def get_sim(r_val=None, n_val=None):
    sim_res = []
    N = n_val
    for r in r_val:
        temp = []
        for run_id in range(1):
            pop = np.ones(shape=[N])
            pop[np.random.choice(range(N), size=N-1, replace=False)] = 0

            count = 0
            while (count < 100000) & (np.sum(pop) > 0) & (np.sum(pop) < N):
                b = np.random.choice(range(N), size=1, p=l_norm([r if x == 1 else 1 for x in pop])).item()
                d = np.random.choice(range(N), size=1).item()
                pop[d] = pop[b]
                count+=1
            temp.append(np.asarray([np.sum(pop), count]))
        sim_res.append(np.stack(temp, axis=0))
    
    return np.stack(sim_res, axis=0)


sim_res = get_sim(r_val=np.arange(1.2,5,0.2), n_val=50)

fix_prob = lambda x: np.sum(x[::,::,0] == 50, axis=1)/x.shape[1]
fix_prob_mat = np.stack([fix_prob(sim_res[::,np.random.choice(np.arange(10000), size=8000),::]) for i in range(100)],axis=1)

# Plot
fig, ax = plt.subplots(figsize=(6,6))
ax.boxplot(fix_prob_mat.T)
ax.set_ylabel('Fixation Probability')
ax.set_xlabel('r')
ax.set_xticks(np.arange(1,20,1))
ax.set_xticklabels(["{:.1f}".format(x) for x in np.arange(1.2,5,0.2)])
plt.savefig('/home/drmz/Desktop/fix_prob.pdf', dpi=600)

def fix_time(x=None):
    temp_ft = []
    for i in range(x.shape[0]):
        local_temp = x[i,]
        local_temp = local_temp[local_temp[::,0]==50,]
        temp_ft.append([np.mean(local_temp[np.random.choice(np.arange(local_temp.shape[0]), size=int(0.8*local_temp.shape[0])),1]) for k in range(100)])
    return np.stack(temp_ft, axis=0)


fix_time_mat = fix_time(sim_res)

fig, ax = plt.subplots(figsize=(6,6))
ax.boxplot(fix_time_mat.T)
ax.set_ylabel('Fixation Time')
ax.set_ylim(100, 100000)
ax.set_yscale('log')
ax.set_xlabel('r')
ax.set_xticks(np.arange(1,20,1))
ax.set_xticklabels(["{:.1f}".format(x) for x in np.arange(1.2,5,0.2)])
plt.savefig('/home/drmz/Desktop/fix_time.pdf', dpi=600)


# Constant Effect
temp = []
for local_n in np.arange(5,100,5):
    local_sim_res = get_sim(r_val = [1.1], n_val=local_n)
    temp.append(local_sim_res.reshape([1000, 2]))

temp = np.stack(temp, axis=0)

fix_prob = lambda x: np.sum(x[::,::,0] > 0, axis=1)/x.shape[1]
fix_prob_pop_mat = np.stack([fix_prob(temp[::,np.random.choice(np.arange(1000), size=800),::]) for i in range(100)],axis=1)

# Plot
fig, ax = plt.subplots(figsize=(6,6))
ax.boxplot(fix_prob_pop_mat.T)
ax.set_ylabel('Fixation Probability')
ax.set_xlabel('Population Size')
ax.set_xticks([0,5,10,15,20])
ax.set_xticklabels([0,25,50,75,100])
plt.savefig('/home/drmz/Desktop/fix_prob_pop.pdf', dpi=600)

def fix_time(x=None):
    temp_ft = []
    for i in range(x.shape[0]):
        local_temp = x[i,]
        local_temp = local_temp[local_temp[::,0]>0,]
        temp_ft.append([np.mean(local_temp[np.random.choice(np.arange(local_temp.shape[0]), size=int(0.8*local_temp.shape[0])),1]) for k in range(100)])
    return np.stack(temp_ft, axis=0)

fix_time_pop_mat = fix_time(temp)

fig, ax = plt.subplots(figsize=(6,6))
ax.boxplot(fix_time_pop_mat.T)
ax.set_ylabel('Fixation Time')
ax.set_ylim(100, 100000)
ax.set_yscale('log')
ax.set_xlabel('Population Time')
ax.set_xticks([0,5,10,15,20])
ax.set_xticklabels([0,25,50,75,100])

plt.savefig('/home/drmz/Desktop/fix_time_pop.pdf', dpi=600)



# Benchmark #
bench_res = []
for n_val in [50,100,250,500,750,1000]:
    local_bench_res = []
    for i in range(100):
        st_time = time.perf_counter()
        get_sim(r_val=[1.1], n_val=n_val)
        end_time = time.perf_counter()
        local_bench_res.append(end_time-st_time)
    bench_res.append(np.stack(local_bench_res))
bench_res = np.stack(bench_res, axis=0)
        

fig, ax = plt.subplots(figsize=(6,6))
ax.boxplot(bench_res.T)
ax.set_ylabel('Runtime (seconds)')
ax.set_xlabel('Population Number')
ax.set_xticklabels([50, 100, 250, 500, 750, 1000])
plt.savefig('/home/drmz/Desktop/sys_time.pdf', dpi=600)

sim_res_combined = []
N = 50

for r in np.arange(1.2,5,0.2):
    sim_res_all = []
    for run_id in range(1000):
        #sim_res = []
        pop = np.ones(shape=[N])
        pop[np.random.choice(range(N), size=49, replace=False)] = 0

        count = 0
        while (count < 100000) & (np.sum(pop) > 0) & (np.sum(pop) < N):
            #sim_res.append(np.copy(pop))
            b = np.random.choice(range(N), size=1, p=l_norm([r if x == 1 else 1 for x in pop])).item()
            d = np.random.choice(range(N), size=1).item()
            pop[d] = pop[b]
            count+=1
        sim_res_all.append(np.copy(pop))
    sim_res_all = np.stack(sim_res_all, axis=0)
    frac_res = np.sum(np.sum(sim_res_all==1, axis=1)==50)/sim_res_all.shape[0]
    sim_res_combined.append(frac_res)
    #sim_res_all.append(frac_res)
#sim_res_all = np.stack(sim_res_all, axis=0)
sim_res_combined = np.stack(sim_res_combined)
np.save('/home/drmz/moran_res_fixation_prob.npy', sim_res_combined)
sim_res_combined = np.load('/home/drmz/moran_res_fixation_prob.npy')

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.arange(1.2,5,0.2), sim_res_combined)
ax.set_ylabel('Fixation Probability')
ax.set_xlabel('r')
plt.savefig('/home/drmz/Desktop/fix_prob.pdf', dpi=600)

fixtime_res_combined=[]
for r in np.arange(1.2,5,0.2):
    fix_res_all = []
    for run_id in range(100):
        #sim_res = []
        pop = np.ones(shape=[N])
        pop[np.random.choice(range(N), size=49, replace=False)] = 0

        count = 0
        while (count < 100000) & (np.sum(pop) > 0) & (np.sum(pop) < N):
            #sim_res.append(np.copy(pop))
            b = np.random.choice(range(N), size=1, p=l_norm([r if x == 1 else 1 for x in pop])).item()
            d = np.random.choice(range(N), size=1).item()
            pop[d] = pop[b]
            count+=1
            #sim_res = np.stack(sim_res, axis=0)
        if np.sum(pop) == N:
            fix_res_all.append(count)
        else:
            fix_res_all.append(0)
    fix_res_all = np.stack(fix_res_all)
    fixtime_res_combined.append(fix_res_all)
fixtime_res_combined = np.stack(fixtime_res_combined, axis=0)
np.save('/home/drmz/moran_res_fixation_time.npy', fixtime_res_combined)

fixtime_res_combined = np.load('/home/drmz/moran_res_fixation_time.npy')
fixtime_res_combined_ft = np.zeros(shape=[19])
for i in range(19):
    temp = fixtime_res_combined[i,::]
    fixtime_res_combined_ft[i] = np.mean(temp[temp>0])
    
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.arange(1.2,5,0.2), fixtime_res_combined_ft)
ax.set_ylabel('Fixation Time')
ax.set_xlabel('r')
plt.savefig('/home/drmz/Desktop/fix_time.pdf', dpi=600)


#sim_res_all = np.stack(sim_res_all, axis=0)
sim_res_combined = np.stack(sim_res_combined)

# Plot
X, Y = np.meshgrid(range(5000), np.arange(0.1,3,0.1))


fig = plt.figure(figsize=(10,8))
ax = fig.gca(projection='3d')
ax.plot_wireframe(X,Y,sim_res_all)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlabel('t')
ax.set_ylabel('r')
ax.set_zlabel('p')

#plt.show()
plt.savefig('/home/drmz/Desktop/m_proc.pdf', dpi=600)