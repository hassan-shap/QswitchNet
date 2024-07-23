from network_utils_hybrid import *
import random
import numpy as np
import time


Nrep = 1
num_gates_list = [10]#np.arange(10,41,10)
num_ToR = 2
n = 4 # must be even, starts from 4

specs = {
    "num_sw_ports": n,
    "num_ToR" : num_ToR,
    "qs_per_node" : 4,
    "bandwidth" : 2,
    "num_bsm_ir" : 2,
    "num_bsm_tel" : 2,
    "num_pd" : 1, # inactive
    "num_laser" : 1, # inactive
    "num_bs" : 1, # inactive
    "num_es" : 1 # inactive
}


telecom_gen_rate = 1/(1e-2) # ebit average generation time in sec
switch_duration = 1e-3 # average switching delay in sec
nir_prob = 1e-2 # NIR gen prob
qubit_reset = 1e-6 # qubit reset time in sec

G, vertex_list = clos_hybrid(specs)
edge_switches, node_list, node_qubit_list =  vertex_list
num_qubits = len(node_qubit_list)

T_tel_list = []
T_nir_list = []
# print(problem_size)

for num_gates in num_gates_list:
    tic = time.time()
    print(num_gates)
    T_tel = []
    T_nir = []
    for _ in range(Nrep):

        connections = []
        for i in range(num_qubits):
            for j in range(i+1,num_qubits):
                if np.random.rand() > 0.5:
                    connections.append((node_qubit_list[i],node_qubit_list[j]))
                else:
                    connections.append((node_qubit_list[j],node_qubit_list[i]))

        # print(node_qubit_list)
        gate_seq = random.choices(connections, k=num_gates)
        switch_seq = network_latency_dag_multiqubit_hybrid(G, vertex_list, gate_seq)
        print(gate_seq)
        print(switch_seq)

    #     tel_latency = []
    #     nir_latency = []
    #     for switch_time in switch_seq:
    #         tel_latency.append(1/telecom_gen_rate * time_spdc(np.array(switch_time)).sum() + switch_duration*len(switch_time))
    #         nir_latency.append(qubit_reset * time_nir(switch_time, nir_prob).sum() + switch_duration*len(switch_time))

    #     T_tel.append(sum(tel_latency) + switch_duration * len(switch_seq))
    #     T_nir.append(sum(nir_latency) + switch_duration * len(switch_seq))
    
    # T_tel_list.append(sum(T_tel)/len(T_tel))
    # T_nir_list.append(sum(T_nir)/len(T_nir))
    toc = time.time()
    print(f"elapsed time {toc-tic} sec")
    
# print(T_tel_list)
# print(T_nir_list)
