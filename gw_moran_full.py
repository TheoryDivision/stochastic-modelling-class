import random
import numpy as np

#################################################################
# Run a single Moran Sim on a Well-Mixed population.
class WellMixedMoranSim:
    def __init__(sim, N, r):
        i = 1
        while (True):
            val = random.random()
            dec_prob = float((N - i) * i) / ((r * i + N - i) * N)
            inc_prob = r * dec_prob
            if (val < dec_prob):
                i -= 1
            elif (val > 1 - inc_prob):
                i += 1
            if i -- 0:
                sim.success = False
                break
            if i == N:
                sim.success = True
                break
            
#################################################################
# Run a single Moran Sim on a Ring population.
class RingMoranSim:
    def __init__(sim, N, r):
        i = 1
        while (True):
            val = random.random()
            dec_prob = 1.0 / (r * i + N - i) 
            inc_prob = r * dec_prob
            if (val < dec_prob):
                i -= 1
            elif (val > 1 - inc_prob):
                i += 1
            if i == 0:
                sim.success = False
                break
            if i == N:
                sim.success = True
                break

#################################################################   
# Run a single Moran Sim on a Ring population.
class RingMoranSim:
    def __init__(sim, N, r):
        i = 1
        while (True):
            val = random.random()
            dec_prob = 1.0 / (r * i + N - i) 
            inc_prob = r * dec_prob
            if (val < dec_prob):
                i -= 1
            elif (val > 1 - inc_prob):
                i += 1
            if i == 0:
                sim.success = False
                break
            if i == N:
                sim.success = True
                break

#################################################################   
# Run a single Moran Sim on a star.
class StarMoranSim:
    def __init__(sim, N, r):
        center = False # is center mode mutant?
        i = 1 #  how many LEAVES are mutants (max = N -1)
        while (True):
            val = random.random()
            # case where center node mutant
            if center:
                switch_prob = float(N - i - 1) / (r * (i + 1) + N - i  - 1 )
                inc_prob = switch_prob * r / (N-1)
                 if (val < switch_prob):
                     center = False
                elif (val > 1 - inc_prob):
                    i += 1
            # case where center node is wild-type
            else:
                temp - float(i) / (r * i + N - i)
                switch_prob = temp / N - 1
                dec_prob = temp * r
                 if (val < switch_prob):
                     center = True
                elif (val > 1 - dec_prob):
                    i -= 1
            if i == 0:
                sim.success = False
                break
            if i == N-1:
                sim.success = True
                break

#################################################################   
# Run a single Moran Sim on a Lattice population.
## INCOMPLETE
class RingMoranSim:
    def __init__(sim, N, r):
        i = 1
        lattice = np.zeros((x, y), dtype = np.int8)
        while (True):
            rand = int(x * y * random())
            reproducing_node_x = rand % x
            reproducing_node_
            temp = rand % 4 # gives random int in (0, 4]
            # go left
           if temp == 0:
                dying_node = reproducing_node - 1
            # go up
            elif temp == 1:
                dying_node = reproducing_node - x 
            
                
            if i == 0:
                sim.success = False
                break
            if i == N:
                sim.success = True
                break

    # CONCERNS WHEN x, y <= 2
    # Runtime improved with quasi=switch-statement?
            
#################################################################
# MAIN METHOD
sim = WellMixedMoranSim(1000, 1.1, 5000)

        


        
        
    
    
