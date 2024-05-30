from network_utils import *
import random
import numpy as np

n = 4 # starts from 4
bandwidth = 2
num_bsm = 2
num_ToR = 2
num_qubits_per_node = 2

gen_rate = 1/(1e-2) # ebit average generation time in sec
switch_duration = 1e-3 # average switching delay in sec

G, vertex_list = clos_multiqubit(n, num_ToR, num_qubits_per_node, bandwidth)
core_switches, agg_switches, edge_switches, node_list, node_qubit_list = vertex_list

problem_size_list = range(1,len(node_qubit_list))
T_list= []
for problem_size in problem_size_list:
    # print(problem_size)
    query_seq, gate_mul_seq = parallel_qft_circuit_gen(node_qubit_list[:problem_size])

    Tvals = network_latency_multiqubit_circuit(G, vertex_list, num_bsm, gen_rate, switch_duration, query_seq, gate_mul_seq, hyperx=False)
    T = np.sum(Tvals) + switch_duration * len(query_seq)
    T_list.append(T)

print(T_list)