import numpy as np
import pandas as pd
import networkx as nx
import sim
import analytic
import timeit
from sim import run_sim

#Fitness range and n range Must be saved in a list or tuple even if only one condition is being run
sim_output = sim.run_sim(
    numsim = 10,
    fitness_range= [1.1],
    n_range = [100],
    init_mut = 1,
    max_time = 5000
    )

math_output = analytic.analytic_solution(n = 50)

analytic_data = pd.DataFrame(math_output)
analytic_data.columns = ["n", "initial_mut", "relative_fitness", "fixation_prob"]
analytic_data.to_csv("C:/Users/dtw43/Documents/Eco_Evo_Research/Stochastic_Practice/data/math_data.csv")
pop_data = pd.DataFrame(sim_output)
pop_data.columns = ["n", "relative_fitness", "sim_num", "time_step", "pop_a", "pop_b"]

pop_data.to_csv("C:/Users/dtw43/Documents/Eco_Evo_Research/Stochastic_Practice/data/sim_data.csv")

#for benchmarking
np.mean(timeit.repeat(
    stmt="run_sim(numsim = 10,fitness_range= [1.1],n_range = [10],init_mut = 1,max_time = 5000)",
    setup = "from sim import run_sim", repeat = 100, number = 1))
#Fuck all this shit I'm going to do this stuff in R
#process the sim output data
#pop_data["relative_fitness"] = pop_data["relative_fitness"].astype(str) #Necessary for plotting purposes
#pop_data = pop_data.assign(
 #   pop_a_normalized = lambda x: x.pop_a/x.n,
 #   time_step_normalized = lambda x: x.time_step/x.n,
 #   relative_fitness = lambda x: "R = " + x.relative_fitness
 #  )

#Group data to grab the mean and standard deviation of the 10 simulations.
""" agg_df = pop_data.groupby(["time_step", "relative_fitness", "n"]).aggregate({
    "time_step_normalized": "first",
    "n": pd.unique,
    "relative_fitness": pd.unique,
    "pop_a_normalized": [np.mean, np.std],
    "pop_b": np.mean
    }, as_index=True)

agg_df.columns = ["_".join(x) for x in agg_df.columns.ravel()]
agg_df = agg_df.droplevel(["relative_fitness", "n"])
agg_df = agg_df.reset_index()

#Have to run the next 3 lines all at once if using the interactive window
fig, ax = plt.subplots(figsize=(8,6))
for label, df in agg_df.groupby(["relative_fitness_unique", "n_unique"]):
    df.plot(kind="line", x = "time_step_normalized_first", y = "pop_a_normalized_mean", ax=ax, label=label)
    plt.fill_between(np.array(df.time_step_normalized_first),
        y1 = np.array(df.pop_a_normalized_mean) - 2* np.array(df.pop_a_normalized_std),
        y2 = np.array(df.pop_a_normalized_mean) + 2* np.array(df.pop_a_normalized_std),
        alpha = 0.2)
plt.legend()
ax.set(ylim = (0,1), xlim = (0,5))
plt.savefig("C:/Users/dtw43/Documents/Eco_Evo_Research/Stochastic_Practice/Moran_Fitness_N_Range.png")

#Now do another plot to show the distribution of termination conditions
agg_df = pop_data.groupby(["relative_fitness", "n", "sim_num"]).aggregate({
    "time_step": np.max,
    "time_step_normalized": np.max})
agg_df = agg_df.reset_index()

fig, ax = plt.subplots(figsize=(16,12))
seaborn.boxplot(
    x = "n", y = "time_step", hue = "relative_fitness",
    data = agg_df
    )
seaborn.swarmplot(x = "n", y = "time_step", hue = "relative_fitness",
    data = agg_df, dodge = True, size = 10)
#Trying to keep the legend from repeating
handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:3], labels[0:3], loc=0)
seaborn.set(font_scale = 3)
plt.ylabel("fixation_time")
plt.xlabel("pop_size")
plt.savefig("C:/Users/dtw43/Documents/Eco_Evo_Research/Stochastic_Practice/Moran_Fixation_Time.png")

 """