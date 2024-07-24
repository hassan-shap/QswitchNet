from network_utils_hybrid import *
import random
import numpy as np
import time
import json

Nrep = 10
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
num_qubits_list = np.arange(50,len(node_qubit_list)+1,50)
print(len(node_qubit_list))

for num_qubits in num_qubits_list:
    print(num_qubits)
    tic = time.time()
    num_gates_list = np.arange(num_qubits,2*num_qubits+1,20)

    connections = []
    for i in range(num_qubits):
        for j in range(i+1,num_qubits):
            if np.random.rand() > 0.5:
                connections.append((node_qubit_list[i],node_qubit_list[j]))
            else:
                connections.append((node_qubit_list[j],node_qubit_list[i]))

    latency_list = []

    for num_gates in num_gates_list:
        # print(num_gates)
        # T_tel = []
        # T_nir = []
        T_latency = []
        for _ in range(Nrep):

            # print(node_qubit_list)
            gate_seq = random.choices(connections, k=num_gates)
            switch_seq = network_latency_dag_multiqubit_hybrid(G, vertex_list, gate_seq)
            switch_seq = np.array(switch_seq)

            tel_latency = 1/telecom_gen_rate * time_spdc(switch_seq[:,1]) 
            nir_latency = qubit_reset * np.array([time_nir[k] for k in switch_seq[:,0]])
            latency_combined = np.stack((tel_latency,nir_latency), axis = 1)

            T_latency.append( np.max(latency_combined, axis = 1).sum() + switch_duration * switch_seq.shape[0] )
        
        latency_list.append(sum(T_latency)/len(T_latency))

    toc = time.time()    
    print(latency_list)
    print(f"elapsed time {toc-tic} sec")
