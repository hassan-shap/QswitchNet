from network_utils import *
import random
import numpy as np
import time
from joblib import Parallel, delayed
import multiprocessing
num_cores = 28                                 

gen_rate = 1/(1e-2) # ebit average generation time in sec
switch_duration = 1e-3 # average switching delay in sec
Nrep = 28 # No. of repetitions for saving separate files
Niter = 200  # number of repetions for ensemble averaging

n_list = [4, 8, 16, 32] # number of core switch ports
lam_gate_list = np.linspace(0.1,0.5,15)  # mean of the Poisson distribution
num_ToR_list = range(2,11,2)

T = np.zeros((len(num_ToR_list),len(lam_gate_list)))
dT = np.zeros((len(num_ToR_list),len(lam_gate_list)))
Nq = np.zeros((len(num_ToR_list),len(lam_gate_list)))

for n in n_list:
    num_node_list =[]
    def runner(i_rep):
        tic = time.time()

        for i_tor, num_ToR in enumerate(num_ToR_list):

            G, vertex_list = clos(n, num_ToR)
            _, _, _, node_list = vertex_list
            num_node = len(node_list)
            num_node_list.append(num_node)

            for i_l, lam_gate_seq in enumerate(lam_gate_list):
                query_seq = np.random.poisson(lam_gate_seq*num_node, Niter)
                query_seq = query_seq[np.argwhere(query_seq>0)[:,0]]
                Nq[i_tor, i_l] = len(query_seq)
                Tvals = network_latency(G, vertex_list, gen_rate, switch_duration, query_seq)
                T[i_tor, i_l] = np.mean(Tvals)
                dT[i_tor, i_l] = np.std(Tvals)

        toc = time.time()

        fname = f"results/network_sim/clos_n_{n}_gen_{1e3/gen_rate:.1f}_sw_{1e3*switch_duration:.1f}_r_{i_rep}.npz"
        print(f"{fname}, elapsed time {toc-tic} sec")
        np.savez(fname, num_ToR_list, lam_gate_list, num_node_list, Nq, T, dT)

    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(Nrep))

print("Finished!")
