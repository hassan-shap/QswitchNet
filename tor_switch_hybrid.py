from network_utils_hybrid import *
import random
import numpy as np
import time

Nrep = 100
num_gates_list = np.arange(10,41,10)
num_ToR = 10

specs = {
    "num_ToR" : num_ToR,
    "qs_per_node" : 10,
    "bandwidth" : 2,
    "num_bsm_ir" : 2,
    "num_pd" : 1, # inactive
    "num_laser" : 1, # inactive
    "num_bs" : 1, # inactive
    "num_es" : 1 # inactive
}

telecom_gen_rate = 1/(1e-2) # ebit average generation time in sec
switch_duration = 1e-3 # average switching delay in sec
nir_prob = 1e-2 # NIR gen prob
qubit_reset = 1e-6 # qubit reset time in sec

G, vertex_list = tor_switch(specs)
edge_switches, node_list, node_qubit_list  = vertex_list

T_tel_list = []
T_nir_list = []
# print(problem_size)

for num_gates in num_gates_list:
    tic = time.time()
    print(num_gates)
    T_tel = []
    T_nir = []
    for _ in range(Nrep):
        query_seq, gate_mul_seq = parallel_circuit_gen(node_list, specs["qs_per_node"], num_gates)
        switch_seq = network_latency_multiqubit_hybrid(G, vertex_list, query_seq, gate_mul_seq)

        tel_latency = []
        nir_latency = []
        for switch_time in switch_seq:
            tel_latency.append(1/telecom_gen_rate * time_spdc(np.array(switch_time)).sum() + switch_duration*len(switch_time))
            nir_latency.append(qubit_reset * time_nir(switch_time, nir_prob).sum() + switch_duration*len(switch_time))

        T_tel.append(sum(tel_latency) + switch_duration * len(switch_seq))
        T_nir.append(sum(nir_latency) + switch_duration * len(switch_seq))
    
    T_tel_list.append(sum(T_tel)/len(T_tel))
    T_nir_list.append(sum(T_nir)/len(T_nir))
    toc = time.time()
    print(f"elapsed time {toc-tic} sec")
    
print(T_tel_list)
print(T_nir_list)
