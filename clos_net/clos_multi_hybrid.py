from network_utils_hybrid import *
import random
import numpy as np
import time
import json

Nrep = 10
num_ToR = 10
n = 4 # must be even, starts from 4
num_bsm_ir = 2
num_bsm_tel = 2
qs_per_node = 10
specs = {
    "num_sw_ports": n,
    "num_ToR" : num_ToR,
    "qs_per_node" : qs_per_node,
    "bandwidth" : 2,
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
# num_jobs_list = range(num_ToR) 
print(len(node_qubit_list))

compute_q_list = []
for job_num in range(num_ToR):
    compute_q_list.append([qs_per_node*num_ToR*j + i + 10*job_num  for j in range(n**2//4) for i in range(qs_per_node)])
# print(compute_q_list)

for num_jobs in range(1,num_ToR+1):
    print(num_jobs)
    tic = time.time()
    num_qubits_per_job = qs_per_node * n**2//4
    num_qubits = num_jobs*num_qubits_per_job
    # num_gates_list = np.linspace(num_qubits_per_job,8*num_qubits_per_job,10)
    num_gates_list =[300]

    connections_all = []
    for i_job in range(num_jobs):
        # print(compute_q_list[i_job][:])
        connections = []
        for i in range(num_qubits_per_job):
            for j in range(i+1,num_qubits_per_job):
                if np.random.rand() > 0.5:
                    connections.append((node_qubit_list[compute_q_list[i_job][i]],node_qubit_list[compute_q_list[i_job][j]]))
                else:
                    connections.append((node_qubit_list[compute_q_list[i_job][j]],node_qubit_list[compute_q_list[i_job][i]]))
        connections_all.append(connections)

    JSON_PATH = f"results/clos_multidag_T_vs_depth/q_{num_jobs}_n_{n}_tor_{num_ToR}_tel_{num_bsm_tel}_nir_{num_bsm_ir}.json"
    with open(JSON_PATH, 'w') as json_file:

        latency_depth_list = []
        for num_gates in num_gates_list:
            num_gates = int(num_gates)

            for _ in range(Nrep):
                gate_seq = []
                for i_job in range(num_jobs):
                    gate_seq += random.choices(connections_all[i_job], k=num_gates)
                # print(gate_seq)

                switch_seq, circ_depth = eff_network_latency_dag_multiqubit_hybrid(G, vertex_list, gate_seq)
                switch_seq = np.array(switch_seq)

                tel_latency = 1/telecom_gen_rate * time_spdc(switch_seq[:,1]) 
                nir_latency = qubit_reset * np.array([time_nir[k] for k in switch_seq[:,0]])
                latency_combined = np.stack((tel_latency,nir_latency), axis = 1)

                T_latency =  np.max(latency_combined, axis = 1).sum() + switch_duration * switch_seq.shape[0] 
                latency_depth_list.append([num_gates,circ_depth, T_latency])

        json_file.write(json.dumps(latency_depth_list) + '\n')


    toc = time.time()    
    # print(latency_list)
    print(f"elapsed time {toc-tic} sec")
