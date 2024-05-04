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
Niter = 100  # number of repetions for ensemble averaging

S = 10 # number of switches for each dim (assuming uniform/regular hyperX network)
L = 3 #len(S) # number of lattice dims
lam_gate_list = np.linspace(0.1,0.5,15)  # mean of the Poisson distribution
num_ToR_list = range(2,11,2)

T = np.zeros((len(num_ToR_list),len(lam_gate_list)))
dT = np.zeros((len(num_ToR_list),len(lam_gate_list)))
Nq = np.zeros((len(num_ToR_list),len(lam_gate_list)))

def runner(i_rep):
    tic = time.time()
    num_node_list = []
    for i_tor, num_ToR in enumerate(num_ToR_list):

        G, vertex_list = hyperX(S, L, num_ToR)
        _, node_list = vertex_list
        num_node = len(node_list)
        num_node_list.append(num_node) 
        # print(num_node)

        for i_l, lam_gate_seq in enumerate(lam_gate_list):
            query_seq = np.random.poisson(lam_gate_seq*num_node, Niter)
            query_seq = query_seq[np.argwhere(query_seq>0)[:,0]]
            Nq[i_tor, i_l] = len(query_seq)
            Tvals = network_latency(G, vertex_list, gen_rate, switch_duration, query_seq, hyperx=True)
            T[i_tor, i_l] = np.mean(Tvals)
            dT[i_tor, i_l] = np.std(Tvals)

    toc = time.time()

    fname = f"results/network_sim/hyperx_{S}_{L}_gen_{1e3/gen_rate:.1f}_sw_{1e3*switch_duration:.1f}_r_{i_rep}.npz"
    print(f"{fname}, elapsed time {toc-tic} sec")
    np.savez(fname, num_ToR_list, lam_gate_list, num_node_list, Nq, T, dT)

results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(Nrep))

print("Finished!")
