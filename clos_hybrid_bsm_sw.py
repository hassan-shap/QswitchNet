from network_utils_hybrid import *
import random
import numpy as np
import time
import json

Nrep = 100
num_ToR = 4
n = 6 # must be even, starts from 4
num_bsm_ir = 2

specs = {
    "num_sw_ports": n,
    "num_ToR" : num_ToR,
    "qs_per_node" : 10,
    "bandwidth" : 2,
    "num_bsm_ir" : num_bsm_ir,
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
num_qubits = num_network_qubits
num_gates = num_qubits
print(len(node_qubit_list))
connections = []
for i in range(num_qubits):
    for j in range(i+1,num_qubits):
        if np.random.rand() > 0.5:
            connections.append((node_qubit_list[i],node_qubit_list[j]))
        else:
            connections.append((node_qubit_list[j],node_qubit_list[i]))


num_bsm_list = np.arange(2,17,2)
num_comm_q_list = [2,4,8]


for num_comm_q in num_comm_q_list:
    specs["bandwidth"] = num_comm_q
    tic = time.time()

    latency_list = []
    # latency_depth_list = []

    JSON_PATH = f"results/bsm_sw/clos_{n}_{num_ToR}_nir_{num_bsm_ir}_comm_{num_comm_q}.json"
    with open(JSON_PATH, 'w') as json_file:

        for num_bsm in num_bsm_list:
            specs["num_bsm_tel"] = num_bsm
            G, vertex_list = clos_hybrid(specs)

            T_latency = []
            # latency_depth = []
            for _ in range(Nrep):
                gate_seq = random.choices(connections, k=num_gates)
                switch_seq, circ_depth = eff_network_latency_dag_multiqubit_hybrid(G, vertex_list, gate_seq)
                switch_seq = np.array(switch_seq)

                tel_latency = 1/telecom_gen_rate * time_spdc(switch_seq[:,1]) 
                nir_latency = qubit_reset * np.array([time_nir[k] for k in switch_seq[:,0]])
                latency_combined = np.stack((tel_latency,nir_latency), axis = 1)

                T_latency.append( np.max(latency_combined, axis = 1).sum() + switch_duration * switch_seq.shape[0] )
                # circ_depths.append(circ_depth)
                # T_latency = np.max(latency_combined, axis = 1).sum() + switch_duration * switch_seq.shape[0]
                # latency_depth.append((T_latency,circ_depth))
            # latency_list.append(sum(T_latency)/len(T_latency))
            # print(T_latency, circ_depths)
            latency_list.append(T_latency)
            # latency_depth_list.append(latency_depth)

        json_file.write(json.dumps(latency_list) + '\n')
        # json_file.write(json.dumps(circ_depths_list) + '\n')

    toc = time.time()    
    # print(latency_list)

    print(f"elapsed time {toc-tic} sec")
