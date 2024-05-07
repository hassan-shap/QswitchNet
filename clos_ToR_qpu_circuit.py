from network_utils import *
import random
import numpy as np
import time
from joblib import Parallel, delayed
import multiprocessing
num_cores = 28                                 

bandwidth = 1
gen_rate = 1/(1e-2) # ebit average generation time in sec
switch_duration = 1e-3 # average switching delay in sec
Nrep = 28 # No. of repetitions for saving separate files
Niter = 10  # number of repetions for ensemble averaging

n_list = [4,8,16] # number of core switch ports
num_gates_list = np.arange(1,5)*100  # mean of the Poisson distribution
num_ToR_list = range(2,11,2)


for n in n_list:
    def runner(i_rep):
        tic = time.time()
        T = np.zeros((len(num_ToR_list),len(num_gates_list)))
        num_node_list = []
        for i_tor, num_ToR in enumerate(num_ToR_list):

            G, vertex_list = clos_multilink(n, num_ToR, bandwidth)
            _, _, _, node_list = vertex_list
            num_node = len(node_list)
            num_node_list.append(num_node) 
            for i_g, num_gates in enumerate(num_gates_list):
                for _ in range(Niter):
                    query_seq, gate_mul_seq = parallel_circuit_gen(node_list, num_gates)
                    Tvals = network_latency_circuit(G, vertex_list, gen_rate, switch_duration, query_seq)
                    T[i_tor, i_g] += np.sum(Tvals) + switch_duration * len(query_seq)

        toc = time.time()
        T /= Niter
        fname = f"results/network_sim/clos_ToR_n_{n}_depth_gen_{1e3/gen_rate:.1f}_sw_{1e3*switch_duration:.1f}_r_{i_rep}.npz"
        print(f"{fname}, elapsed time {toc-tic} sec")
        np.savez(fname, num_ToR_list, num_gates_list, num_node_list, T)

    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(Nrep))

print("Finished!")
