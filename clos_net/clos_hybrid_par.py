from network_utils_hybrid import *
import random
import numpy as np
import time
import json
from joblib import Parallel, delayed
import multiprocessing
num_cores = 28                                 

Nrep = 28 # No. of repetitions for saving separate files
Niter = 10  # number of repetions for ensemble averaging

num_ToR = 4
n = 6 # must be even, starts from 4
qs_per_node = 10
num_bsm_ir = 2
num_bsm_tel = 2
bandwidth = 2

specs = {
    "num_sw_ports": n,
    "num_ToR" : num_ToR,
    "qs_per_node" : qs_per_node,
    "bandwidth" : bandwidth,
    "num_bsm_ir" : num_bsm_ir,
    "num_bsm_tel" : num_bsm_tel,
    "num_pd" : 1, # inactive
    "num_laser" : 1, # inactive
    "num_bs" : 1, # inactive
    "num_es" : 1 # inactive
}

JSON_PATH = "data/nir_latency.json"
with open(JSON_PATH) as f:
    time_nir = np.array(json.load(f))
    # print(time_nir)

telecom_gen_rate = 1/(1e-2) # ebit average generation time in sec
switch_duration = 1e-3 # average switching delay in sec
nir_prob = 1e-2 # NIR gen prob
qubit_reset = 1e-6 # qubit reset time in sec

G, vertex_list = clos_hybrid(specs)
edge_switches, node_list, node_qubit_list =  vertex_list
num_network_qubits = len(node_qubit_list)
num_qubits_list = np.arange(80,len(node_qubit_list)+1,num_ToR*qs_per_node)
# num_qubits_list = [120]
print(len(node_qubit_list))

for num_qubits in num_qubits_list:
    print(num_qubits)
    tic = time.time()
    # num_gates_list = np.arange(num_qubits,20*num_qubits+1,100)
    num_gates_list = np.linspace(num_qubits,30*num_qubits,10)

    connections = []
    for i in range(num_qubits):
        for j in range(i+1,num_qubits):
            if np.random.rand() > 0.5:
                connections.append((node_qubit_list[i],node_qubit_list[j]))
            else:
                connections.append((node_qubit_list[j],node_qubit_list[i]))

    def runner(i_rep):
        tic = time.time()

        latency_depth_list = []
        for num_gates in num_gates_list:
            num_gates = int(num_gates)
            # print(num_gates)
            # T_tel = []
            # T_nir = []
            # T_latency = []
            # latency_depth = []
            for _ in range(Niter):

                # print(node_qubit_list)
                gate_seq = random.choices(connections, k=num_gates)
                switch_seq, circ_depth = eff_network_latency_dag_multiqubit_hybrid(G, vertex_list, gate_seq)
                switch_seq = np.array(switch_seq)

                tel_latency = 1/telecom_gen_rate * time_spdc(switch_seq[:,1]) 
                nir_latency = qubit_reset * np.array([time_nir[k] for k in switch_seq[:,0]])
                latency_combined = np.stack((tel_latency,nir_latency), axis = 1)

                T_latency =  np.max(latency_combined, axis = 1).sum() + switch_duration * switch_seq.shape[0] 
                latency_depth_list.append([int(num_gates), circ_depth, T_latency])
                # latency_depth.append((circ_depth, T_latency))

            # latency_depth_list.append(latency_depth)
        JSON_PATH = f"results/clos_T_vs_depth/q2_{num_qubits}_n_{n}_tor_{num_ToR}_tel_{num_bsm_tel}_nir_{num_bsm_ir}_r_{i_rep}.json"
        with open(JSON_PATH, 'w') as json_file:
            json_file.write(json.dumps(latency_depth_list) + '\n')

        toc = time.time()    
        # print(latency_list)
        print(f"elapsed time {toc-tic} sec")

    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(Nrep))

print("Finished!")