import numpy as np
import random
import networkx as nx


def time_spdc(k_list):
    harmonic = np.zeros(len(k_list))
    for i_k, k in enumerate(k_list):
        harmonic[i_k] = (1/np.arange(1,k+1)).sum()
    return harmonic

# time_spdc = lambda k: (1/np.arange(1,k+1)).sum()

def network_latency(G, vertex_list, gen_rate, switch_duration, query_seq, hyperx=False):
    # print(query_seq)
    if hyperx:
        edge_switches, node_list = vertex_list
    else:
       core_switches, agg_switches, edge_switches, node_list = vertex_list
    num_nodes = len(node_list)
    num_edge = len(edge_switches)
    latency = np.zeros(len(query_seq))
    for i_q, query in enumerate(query_seq):
        if 2*query < num_nodes:
            g_node = random.sample(range(num_nodes), 2*query)
        else:
            # print("hi")
            g_node = np.arange(num_nodes)
            np.random.shuffle(g_node)
        # gate_seq = [(g_node[2*i],g_node[2*i+1]) for i in range((len(g_node)-1)//2+1)]
        gate_seq = [(node_list[g_node[2*i]],node_list[g_node[2*i+1]]) for i in range((len(g_node)-1)//2+1)]
        # print(query, "gate seq:", gate_seq)

        gate_seq_iter = gate_seq.copy()

        switch_time = []
        while len(gate_seq_iter)>0:
            bsm_stat = np.zeros(num_edge,dtype=np.int16)
            G_ins =  G.copy()
            # print(gate_seq_iter)
            inds_keep = []
            for i_g, g in enumerate(gate_seq_iter):
                # print(g)
                if nx.has_path(G_ins,g[0],g[1]):
                    shortestpath = nx.shortest_path(G_ins,g[0],g[1])
                    # print(shortestpath)
                    
                    sp = []
                    for i in range(0,len(shortestpath)-1):
                        sp.append((shortestpath[i],shortestpath[i+1]))
                    
                    b = []
                    for i, edge in enumerate(edge_switches):
                        if edge in shortestpath:
                            b.append(i)
                   
                    if len(b)>1:
                        if bsm_stat[b[0]] == 0 and bsm_stat[b[1]] == 0:
                            bsm_stat[random.sample(b,1)] = 1
                            for u, v in sp:
                                G_ins.remove_edge(u, v)
                        elif bsm_stat[b[0]] == 0:
                            bsm_stat[b[0]] = 1
                            for u, v in sp:
                                G_ins.remove_edge(u, v)
                        elif bsm_stat[b[1]] == 0:
                            bsm_stat[b[1]] = 1
                            for u, v in sp:
                                G_ins.remove_edge(u, v)
                    elif bsm_stat[b] == 0:
                        bsm_stat[b] = 1
                        for u, v in sp:
                            G_ins.remove_edge(u, v)

                else:
                    inds_keep.append(i_g)

            switch_time.append(np.array(bsm_stat).sum())
            gate_seq_iter = [gate_seq_iter[idx] for idx in inds_keep]

        # print("num seq:", len(gate_seq), switch_time)
        latency[i_q] = 1/gen_rate*time_spdc(np.array(switch_time)).sum() + switch_duration*len(switch_time)
    return latency


# network topology generating functions

def clos(n, num_ToR):
    """"
    inputs:
    n : even integer, number of switch ports
    num_ToR: number of top of the rack ports (nodes per rack)
    outputs:
    G : network graph
    vertex_list: list of indices corresponding to
    core_switches, agg_switches, edge_switches, and node_list
    respectively.
    """

    # n = 8 # starts from 4
    num_core = n // 2
    num_agg = n
    num_edge = n**2 // 4
    # num_ToR = 3
    num_nodes = num_edge * num_ToR # number of q nodes
    # num_bsms = num_leaves # number of BSMs

    # if n==4:
    #     conn_right = [7]
    #     conn_left = [8]
    # elif n==6:
    #     conn_right = [11,14]
    #     conn_left = [12,15]
    # elif n==8:
    #     conn_right = [15,19,23]
    #     conn_left = [16,20,24]

    num_vertices = num_core + num_agg + num_edge + num_nodes

    G = nx.Graph()
    core_switches = range(num_core)
    G.add_nodes_from(core_switches, type='core')
    agg_switches = range(num_core,num_core+num_agg)
    G.add_nodes_from(agg_switches, type='agg')
    edge_switches = range(num_core+num_agg,num_core + num_agg + num_edge)
    G.add_nodes_from(edge_switches, type='agg')
    node_list = range(num_core + num_agg + num_edge,num_vertices)
    G.add_nodes_from(node_list, type='node')

    for core in core_switches:
        for agg in agg_switches:
            G.add_edge(core,agg)

    # new_edges = []
    # extra_edges = []
    agg_conn = np.ones(num_agg)* (n//2)
    for i, edge in enumerate(edge_switches):
        i1 = np.argwhere(agg_conn>0)[0,0]
        G.add_edge(edge,agg_switches[i1])
        agg_conn[i1] -= 1 
        # if edge in conn_left:
        #     extra_edges.append((edge,agg_switches[i1]))
        #     new_edges.append((edge,agg_switches[i1-1]))
        G.add_edge(edge,agg_switches[i1+1])
        agg_conn[i1+1] -= 1 
    #     if edge in conn_right:
    #         extra_edges.append((edge,agg_switches[i1+1]))
    #         new_edges.append((edge,agg_switches[i1+2]))

    # G.remove_edges_from(extra_edges)
    # G.add_edges_from(new_edges)

    for i, edge in enumerate(edge_switches):
        for j in range(num_ToR):
            G.add_edge(edge,node_list[num_ToR*i+j])

    vertex_list = core_switches, agg_switches, edge_switches, node_list
    return G, vertex_list


def fat_tree(n):
    """"
    inputs:
    n : even integer, number of switch ports
    outputs:
    G : network graph
    vertex_list: list of indices corresponding to
    core_switches, agg_switches, edge_switches, and node_list
    respectively.
    """
    num_core = n**2 // 4
    num_agg = n * (n // 2)
    num_edge = n * (n // 2)
    num_nodes = num_edge * (n // 2) # number of q nodes

    num_vertices = num_core + num_agg + num_edge + num_nodes

    G = nx.Graph()
    core_switches = range(num_core)
    G.add_nodes_from(core_switches, type='core')
    agg_switches = range(num_core,num_core+num_agg)
    G.add_nodes_from(agg_switches, type='agg')
    edge_switches = range(num_core+num_agg,num_core + num_agg + num_edge)
    G.add_nodes_from(edge_switches, type='agg')
    node_list = range(num_core + num_agg + num_edge,num_vertices)
    G.add_nodes_from(node_list, type='node')

    for i_c, core in enumerate(core_switches):
        for i_a, agg in enumerate(agg_switches):
            if i_c % 2 ==0 and i_a % 2 ==0:
                G.add_edge(core,agg)
            if i_c % 2 ==1 and i_a % 2 ==1:
                G.add_edge(core,agg)

    for i_a, agg in enumerate(agg_switches):
        G.add_edge(agg,edge_switches[i_a])
        if i_a % 2 == 0:
            G.add_edge(agg,edge_switches[i_a+1])
        else:
            G.add_edge(agg,edge_switches[i_a-1])

    for i, edge in enumerate(edge_switches):
        for j in range(n//2):
            G.add_edge(edge,node_list[(n//2)*i+j])

    vertex_list = core_switches, agg_switches, edge_switches, node_list
    return G, vertex_list

def hyperX(S,L,num_ToR):
    # S = 3 # number of switches for each dim (assuming uniform/regular hyperX network)
    # L = 2 #len(S) # number of lattice dims
    # K = 1 # link bandwidth

    # num_ToR = 4 # T: number of terminals (nodes per rack)
    num_switches = S**L
    num_nodes =  num_switches * num_ToR

    G = nx.Graph()
    edge_switches = [] #np.zeros(num_switches,L)
    for sw in range(num_switches):
        result = ""
        for i_l in range(L):
            if sw > 0:
                remainder = sw % S
                result = f"{remainder}," + result
                sw //= S
            else:
                result = "0," + result

        edge_switches.append(result[:-1])

    G.add_nodes_from(edge_switches, type='switch')

    node_list = range(1,num_nodes+1)
    G.add_nodes_from(node_list, type='node')

    for i, sw in enumerate(edge_switches):
        for j in range(num_ToR):
            G.add_edge(sw,node_list[num_ToR*i+j])

    for i1, sw1 in enumerate(edge_switches):
        sw_vec = sw1.split(",")
        for i_l in range(L):
            sw_vals = list(set(range(S))-{int(sw_vec[i_l])})
            for i_s in sw_vals:
                sw2 = sw_vec[:]
                sw2[i_l]= f"{i_s}"
                
                G.add_edge(sw1,",".join(sw2))

    vertex_list = edge_switches, node_list
    return G, vertex_list
