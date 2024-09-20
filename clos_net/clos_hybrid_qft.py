from network_utils_hybrid import *
import random
import numpy as np
import time
import json

num_ToR = 4
n = 6 # must be even, starts from 4

specs = {
    "num_sw_ports": n,
    "num_ToR" : num_ToR,
    "qs_per_node" : 10,
    "bandwidth" : 2,
    "num_bsm_ir" : 2,
    "num_bsm_tel" : 2,
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
# num_qubits_list = np.arange(50,len(node_qubit_list)+1,10)
num_qubits_list = np.arange(20,201+1,20)
print(len(node_qubit_list))

latency_list = []

for num_qubits in num_qubits_list:
    print(num_qubits)
    tic = time.time()

    gate_seq = []
    for target in range(num_qubits-1,0,-1):
        for control in range(target-1,-1,-1):
            gate_seq.append((node_qubit_list[control],node_qubit_list[target]))
    switch_seq = network_latency_dag_multiqubit_hybrid(G, vertex_list, gate_seq)
    switch_seq = np.array(switch_seq)

    tel_latency = 1/telecom_gen_rate * time_spdc(switch_seq[:,1]) 
    nir_latency = qubit_reset * np.array([time_nir[k] for k in switch_seq[:,0]])
    latency_combined = np.stack((tel_latency,nir_latency), axis = 1)

    T_latency =  np.max(latency_combined, axis = 1).sum() + switch_duration * switch_seq.shape[0] 
    
    latency_list.append(T_latency)

    toc = time.time()    
    print(f"elapsed time {toc-tic} sec")
print(latency_list)
