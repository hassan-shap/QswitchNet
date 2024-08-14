from network_utils_hybrid import *
import random
import numpy as np
import time
import json
from joblib import Parallel, delayed
import multiprocessing
num_cores = 28                                 

Nrep = 28 # No. of repetitions for saving separate files
Niter = 100  # number of repetions for ensemble averaging

num_ToR = 4
n = 6 # must be even, starts from 4
qs_per_node = 10
num_bsm_ir = 2
num_bsm_tel = 2
bandwidth = 2


telecom_gen_rate = 1/(1e-2) # ebit average generation time in sec
switch_duration = 1e-3 # average switching delay in sec
nir_prob = 1e-2 # NIR gen prob
qubit_reset = 1e-6 # qubit reset time in sec
comm_qs = 4

specs = {
    "num_sw_ports": n,
    "num_ToR" : num_ToR,
    "qs_per_node" : qs_per_node,
    "bandwidth" : comm_qs,
    "num_bsm_ir" : num_bsm_ir,
    "num_bsm_tel" : num_bsm_tel,
    "telecom_gen_rate" : telecom_gen_rate,
    "switch_duration" : switch_duration,
    "num_pd" : 1, # inactive
    "num_laser" : 1, # inactive
    "num_bs" : 1, # inactive
    "num_es" : 1 # inactive
}

# JSON_PATH = "data/nir_latency.json"
# with open(JSON_PATH) as f:
#     time_nir = np.array(json.load(f))
#     # print(time_nir)

req_rate_list = np.logspace(-2,1.0,10)
num_tel_bsm_list = [4] #np.arange(6,15,2)

# num_ToR_list = np.arange(2,11,2)
# for num_ToR in num_ToR_list:
#     print(f"num_ToR = {num_ToR}")
for num_bsm_tel in num_tel_bsm_list:
    print(f"num_bsm = {num_bsm_tel}")
    specs["num_bsm_tel"] = num_bsm_tel
    G, vertex_list = clos_hybrid(specs)
    edge_switches, node_list, node_qubit_list =  vertex_list
    num_network_qubits = len(node_qubit_list)
    # print(num_network_qubits)

    def runner(i_rep):
        tic = time.time()
        job_duration_wait_list = []
        for i_req, req_rate in enumerate(req_rate_list):
            total_time = Niter/req_rate
            arrival_times = poisson_random_process(req_rate,total_time)
            start_finish_times, circ_depth_list  = clos_job_scheduler(specs, G, vertex_list, arrival_times)

            mean_compute = (start_finish_times[num_ToR:-1,1] - start_finish_times[num_ToR:-1,0]).mean()
            mean_wait = (start_finish_times[num_ToR:-1,0] - arrival_times[num_ToR:-1]).mean()
            job_duration_wait_list.append([mean_compute, mean_wait])

        fname = f"results/clos_multiten/q_N_{Niter}_n_{n}_tor_{num_ToR}_tel_{num_bsm_tel}_comm_{comm_qs}_r_{i_rep}.json"
        with open(fname, 'w') as json_file:
            json_file.write(json.dumps(job_duration_wait_list) + '\n')

        toc = time.time()    
        print(f"{i_rep}, elapsed time {toc-tic} sec")

    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(Nrep))

print("Finished!")