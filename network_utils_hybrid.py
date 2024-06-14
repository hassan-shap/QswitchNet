import numpy as np
import random
import networkx as nx


def time_spdc(k_list):
    harmonic = np.zeros(len(k_list))
    for i_k, k in enumerate(k_list):
        harmonic[i_k] = (1/np.arange(1,k+1)).sum()
    return harmonic

F_geo = lambda x,p: np.floor(np.log(1-x)/np.log(1-p))

def time_nir(switch_time, p, Nmax =  int(1e6)):
    Nmax = int(Nmax) # to make sure Nmax is an integer
    latency = np.zeros(len(switch_time))
    for i_N, Nlinks in enumerate(switch_time):
        Ns = np.zeros((Nlinks, Nmax))
        for i in range(Nlinks):
            Ns[i,:] = F_geo(np.random.uniform(low=p, high=1, size=(Nmax,)),p)

        latency[i_N] = np.mean(np.max(Ns,axis=0))
    return latency

def network_latency_multiqubit_hybrid(G, vertex_list, query_seq, gate_mul_seq):

    edge_switches, node_list, node_qubit_list = vertex_list
    num_nodes = len(node_list)
    num_edge = len(edge_switches)
    switch_seq = []
    for i_q, gate_seq in enumerate(query_seq):
        
        gate_seq_iter = gate_seq[:]
        gate_mul_seq_iter = gate_mul_seq[i_q]

        switch_time = []

        while len(gate_seq_iter)>0:
            num_ir_swap = 0
            num_tel_swap = 0
            G_ins =  G.copy()

            inds_keep = []
            for i_g, g in enumerate(gate_seq_iter):
                for link in range(gate_mul_seq_iter[g]):
                    n0 = g[0]
                    n1 = g[1]
                    if nx.has_path(G_ins,n0,n1):
                        paths = nx.all_shortest_paths(G_ins, n0, n1, weight=None)
                        path_found = False
                        for shortestpath in paths:
                            tel_ir = "ir"
                            if len(shortestpath)> 5 :
                                tel_ir = "tel"

                            sp = []
                            b = []
                            for i in range(0,len(shortestpath)-1):
                                sp.append((shortestpath[i],shortestpath[i+1]))
                                if 1 < i < len(shortestpath)-2:
                                    sw = shortestpath[i]
                                    if G_ins.nodes[sw]["BSM_"+tel_ir] > 0:
                                        b.append(sw)
                            
                            if len(b)>=1:
                                sw_bsm = random.sample(b,1)[0]
                                G_ins.nodes[sw_bsm]["BSM_"+tel_ir]-= 1
                                for u, v in sp:
                                    if G_ins[u][v]['weight'] == 1:
                                        G_ins.remove_edge(u, v)
                                    else:
                                        G_ins[u][v]['weight'] -= 1
                                if  tel_ir == "tel":
                                    num_tel_swap += 1
                                else:
                                    num_ir_swap += 1
                                
                                path_found = True
                                break
            
                        if not path_found:
                            inds_keep.append(i_g)
                            gate_mul_seq_iter[g] -= link
                            break
                    else:
                        inds_keep.append(i_g)
                        gate_mul_seq_iter[g] -= link
                        break

            switch_time.append([num_ir_swap, num_tel_swap])
            # switch_time.append(num_ir_swap)

            gate_seq_iter = [gate_seq_iter[idx] for idx in inds_keep]
            gate_mul_seq_iter = {g:gate_mul_seq_iter[g] for g in gate_seq_iter}
   
        switch_seq.append(switch_time)
        # 1/gen_rate*time_spdc(np.array(switch_time)).sum() + switch_duration*len(switch_time)
    return switch_seq


def parallel_circuit_gen(node_list, qubits_per_node, num_gates):
    num_nodes = len(node_list)
    # node_list = range(num_nodes)
    # qubits_per_node = 3
    qubit_list = range(qubits_per_node)
    # num_qubits = num_nodes * qubits_per_node
    # num_gates = 30

    node_qubit_list = []
    for node in node_list:
        for qubit in range(qubits_per_node):
            node_qubit_list.append((f"{node},{qubit}"))

    connections = []
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            connections.append((node_list[i],node_list[j]))

    gate_seq_nodes = random.choices(connections, k=num_gates)
    # print(gate_seq_nodes)
    gate_seq = []
    for n1, n2 in gate_seq_nodes:
        gate_seq.append((f"{n1},{random.sample(qubit_list,1)[0]}",f"{n2},{random.sample(qubit_list,1)[0]}"))

    gate_seq_iter = gate_seq.copy()

    Q = nx.Graph()
    Q.add_nodes_from(node_qubit_list)

    query_seq = []
    query = []
    gate_mul = {}
    gate_mul_seq = []
    while len(gate_seq_iter)>0:
        # print(gate_seq_iter)
        inds_keep = []
        not_block_gate = True
        for i_g, gate_nodes in enumerate(gate_seq_iter):
            if Q.degree[gate_nodes[0]] > 0 or Q.degree[gate_nodes[1]] > 0:
                if gate_nodes in query and not_block_gate:
                    gate_mul[gate_nodes] += 1
                    # query.append(gate_nodes)
                else:
                    Q.add_edge(gate_nodes[0],gate_nodes[1])
                    inds_keep.append(i_g)
                    not_block_gate = False
            else:
                Q.add_edge(gate_nodes[0],gate_nodes[1])
                query.append(gate_nodes)
                gate_mul[gate_nodes] =  1
                not_block_gate = True

        query_seq.append(query)
        gate_mul_seq.append(gate_mul)
        query = []
        gate_mul = {}
        Q = nx.Graph()
        Q.add_nodes_from(node_qubit_list)
        gate_seq_iter = [gate_seq_iter[idx] for idx in inds_keep]


    return query_seq, gate_mul_seq

def tor_switch(specs):
    num_ToR = specs["num_ToR"]
    num_qubits_per_node = specs["qs_per_node"]
    num_edge = 1
    num_nodes = num_edge * num_ToR # number of q nodes

    num_bsm_edge = specs["num_bsm_ir"]
    num_pd = specs["num_pd"]
    num_laser = specs["num_laser"]
    num_bs = specs["num_bs"]
    num_es = specs["num_es"]

    bandwidth = specs["bandwidth"]
    edge_bw = bandwidth
    num_vertices = num_edge + num_nodes

    G = nx.Graph()
    ## adding node attributes
    # "PD", "BSM", "Laser", "BS", "ES"
    attrs = {}

    edge_switches = range(num_edge)
    G.add_nodes_from(edge_switches, type='edge')
    for edge in edge_switches:
        attrs[edge] = {"PD": num_pd, "BSM_ir": num_bsm_edge, "BSM_tel":0, "Laser": num_laser, "BS": num_bs, "ES": num_es}
    node_list = range(num_edge,num_vertices)
    G.add_nodes_from(node_list, type='node')
    node_qubit_list = []
    for node in node_list:
        for qubit in range(num_qubits_per_node):
            qname = f"{node},{qubit}"
            node_qubit_list.append(qname)
            G.add_edge(node,qname, weight=1)
            
    nx.set_node_attributes(G, attrs)

    for i, edge in enumerate(edge_switches):
        for j in range(num_ToR):
            G.add_edge(edge,node_list[num_ToR*i+j], weight=edge_bw)

    vertex_list = edge_switches, node_list, node_qubit_list 
    return G, vertex_list

def clos_hybrid(specs):
    n = specs["num_sw_ports"] # starts from 4
    num_ToR = specs["num_ToR"]
    num_qubits_per_node = specs["qs_per_node"]

    num_bsm_edge = specs["num_bsm_ir"]
    num_bsm_agg = specs["num_bsm_tel"]
    num_pd = specs["num_pd"]
    num_laser = specs["num_laser"]
    num_bs = specs["num_bs"]
    num_es = specs["num_es"]

    bandwidth = specs["bandwidth"]

    num_core = n // 2
    num_agg = n
    num_edge = n**2 // 4
    num_nodes = num_edge * num_ToR # number of q nodes
    # num_bsms = num_leaves # number of BSMs

    num_vertices = num_core + num_agg + num_edge + num_nodes
    core_bw = 4*bandwidth
    agg_bw = 2*bandwidth
    edge_bw = bandwidth

    G = nx.Graph()
    ## adding node attributes
    # "PD", "BSM", "Laser", "BS", "ES"
    attrs = {}

    core_switches = range(num_core)
    G.add_nodes_from(core_switches, type='core')
    for core in core_switches:
        attrs[core] = {"PD": 0, "BSM_ir":0, "BSM_tel":0, "Laser":0, "BS":0, "ES":0}
    agg_switches = range(num_core,num_core+num_agg)
    G.add_nodes_from(agg_switches, type='agg')
    for agg in agg_switches:
        attrs[agg] = {"PD": 0, "BSM_ir":0, "BSM_tel":num_bsm_agg, "Laser":0, "BS":0, "ES":0}
    edge_switches = range(num_core+num_agg,num_core + num_agg + num_edge)
    G.add_nodes_from(edge_switches, type='edge')
    for edge in edge_switches:
        attrs[edge] = {"PD": num_pd, "BSM_ir": num_bsm_edge, "BSM_tel":0, "Laser": num_laser, "BS": num_bs, "ES": num_es}
    node_list = range(num_core + num_agg + num_edge,num_vertices)
    G.add_nodes_from(node_list, type='node')
    node_qubit_list = []
    for node in node_list:
        for qubit in range(num_qubits_per_node):
            qname = f"{node},{qubit}"
            node_qubit_list.append(qname)
            G.add_edge(node,qname, weight=1)
            
    nx.set_node_attributes(G, attrs)
    for core in core_switches:
        for agg in agg_switches:
            G.add_edge(core,agg, weight=core_bw)

    agg_conn = np.ones(num_agg)* (n//2)
    for i, edge in enumerate(edge_switches):
        i1 = np.argwhere(agg_conn>0)[0,0]
        G.add_edge(edge,agg_switches[i1], weight=agg_bw)
        agg_conn[i1] -= 1 
        G.add_edge(edge,agg_switches[i1+1], weight=agg_bw)
        agg_conn[i1+1] -= 1 

    for i, edge in enumerate(edge_switches):
        for j in range(num_ToR):
            G.add_edge(edge,node_list[num_ToR*i+j], weight=edge_bw)

    vertex_list = edge_switches, node_list, node_qubit_list 
    return G, vertex_list

