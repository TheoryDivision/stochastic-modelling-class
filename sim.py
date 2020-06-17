import numpy as np
choice = np.random.choice
class sim_env:
    def __init__(self,a,b):
        self.num_a = a
        self.num_b = b
       #self.pop_data = [[0, self.num_a, self.num_b]]
    def update(self, R = 1, N = 1000): #R is the relative fitness of a vs. b
        i = self.num_a
        birth_a_prob = (R*i/(R*i + (N - i))) * ((N - i)/N)
        death_a_prob = ((N - i)/(R * i + (N-i))) * (i/N)
        birth_obj = choice(("a", "b"), 1, p = [birth_a_prob, 1-birth_a_prob])
        if birth_obj == "a":
            self.num_a += 1
        else:
            self.num_b += 1
        death_obj = choice(("a", "b"), 1, p = [death_a_prob, 1-death_a_prob])
        if death_obj == "a":
            self.num_a -= 1
        else:
            self.num_b -= 1

#function for running the simulation - makes use of the sim_env class.
def run_sim(numsim = 10, fitness_range = [1], n_range = [1000], init_mut = 1, max_time = 5000):
    sim_output = []
    for r in fitness_range:
        for n in n_range:
            for i in range(1, numsim):
                new_sim = sim_env(a = init_mut, b = n-init_mut)
                time_step = 0
                while new_sim.num_a > 0 and new_sim.num_b > 0 and time_step <= max_time:
                    time_step += 1
                    new_sim.update(R = r, N = n)
                    sim_output.append([n, r, i, time_step, new_sim.num_a, new_sim.num_b])
    return sim_output