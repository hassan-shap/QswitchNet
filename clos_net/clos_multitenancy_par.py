from network_utils_hybrid import *
import random
import numpy as np
import time
import json
from joblib import Parallel, delayed
import multiprocessing
num_cores = 28                                 

Nrep = 28 * 2 # No. of repetitions for saving separate files
# Niter = 20  # number of repetions for ensemble averaging

num_ToR = 4
n = 6 # must be even, starts from 4
qs_per_node = 10
num_bsm_ir = 2
num_bsm_tel = 2


telecom_gen_rate = 1/(1e-2) # ebit average generation time in sec
switch_duration = 1e-3 # average switching delay in sec
nir_prob = 1e-2 # NIR gen prob
qubit_reset = 1e-6 # qubit reset time in sec
comm_qs = 4
buffer = 4

specs = {
    "buffer_size": buffer,
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

# req_rate_list = np.logspace(-1.7,1.0,10)
req_rate_list = np.logspace(-1.4,1.0,10)
num_tel_bsm_list = [4]#np.arange(8,15,2)
Niter0 = 200
Niter_list = Niter0 * np.linspace(1, 5, len(req_rate_list))

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
        job_list = {"exec": [] ,"completion": [], "reject": [], "qpu_usage": []}
        for i_req, req_rate in enumerate(req_rate_list):
            Niter = Niter_list[i_req]
            total_time = Niter/req_rate
            arrival_times = poisson_random_process(req_rate,total_time)
            time_sum, rej_vec, qpu_usage, circ_depth_list  = clos_job_scheduler_buffer(specs, G, vertex_list, arrival_times)

            job_list["exec"].append(time_sum[0])
            job_list["completion"].append(time_sum[1])
            job_list["reject"].append(rej_vec)
            job_list["qpu_usage"].append(qpu_usage)

        fname = f"q_N_{Niter0}_buff_{buffer}_n_{n}_tor_{num_ToR}_tel_{num_bsm_tel}_comm_{comm_qs}_r_{i_rep}.json"
        print(fname)
        fname_path = "results/clos_multiten/" + fname
        with open(fname_path, 'w') as json_file:
            json_file.write(json.dumps(job_list) + '\n')

        toc = time.time()    
        print(f"{i_rep}, elapsed time {toc-tic} sec")
    # runner(0)
    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(Nrep))

print("Finished!")