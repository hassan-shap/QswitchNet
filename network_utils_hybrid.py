import numpy as np
import random
import networkx as nx
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
import json

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

def poisson_random_process(lmbda, total_time):
    """
    Generate a Poisson random process.
    
    Parameters:
        lmbda (float): The parameter lambda of the Poisson distribution.
        total_time (float): The total time for the process.
    
    Returns:
        numpy.ndarray: An array of timestamps when events occur.
    """
    num_events = np.random.poisson(lmbda * total_time)
    event_times = np.cumsum(np.random.exponential(1 / lmbda, num_events))
    event_times = event_times[event_times < total_time]
    return event_times-event_times[0]

def construct_dag(node_qubit_list, q_assignment, num_gates):
    node_qubit_names = [node_qubit_list[q] for q in q_assignment]
    # print(node_qubit_names)
    qubit_nx_to_qiskit = {qubit: idx for idx, qubit in enumerate(node_qubit_names)}
    num_qubits = len(q_assignment)
    connections = []
    for i in range(num_qubits):
        for j in range(i+1,num_qubits):
            if np.random.rand() > 0.5:
                connections.append((node_qubit_names[i],node_qubit_names[j]))
            else:
                connections.append((node_qubit_names[j],node_qubit_names[i]))

    gate_seq = random.choices(connections, k=num_gates)
    qiskit_q_list = QuantumRegister(num_qubits, "q")
    circ = QuantumCircuit(qiskit_q_list)
    for g in gate_seq:
        q1 = qubit_nx_to_qiskit[g[0]]
        q2 = qubit_nx_to_qiskit[g[1]]
        circ.cx(q1,q2)
    dag = circuit_to_dag(circ)
    circ_depth = dag.depth()
    dag_qubit_map = {bit: index for index, bit in enumerate(dag.qubits)}
    # print(gate_seq)
    # print(circ)
    return dag, circ_depth, dag_qubit_map

def clos_job_scheduler_qpu(specs, G, vertex_list, arrival_times, qpu_reqs):
    JSON_PATH = "data/nir_latency.json"
    with open(JSON_PATH) as f:
        time_nir = np.array(json.load(f))
        # print(time_nir)

    buffer = specs["buffer_size"]

    n = specs["num_sw_ports"]
    telecom_gen_rate = specs["telecom_gen_rate"]
    qubit_reset = specs["qubit_reset"]
    switch_duration = specs["switch_duration"]
    num_ToR = specs["num_ToR"]
    qs_per_node = specs["qs_per_node"]
    qpu_vals = specs["qpu_vals"]
    _, _, node_qubit_list = vertex_list

    num_jobs = len(arrival_times)
    # print(num_jobs)
    # qpu_vals = [2,3,4,5,6]
    # qpu_reqs = np.random.choice(qpu_vals, num_jobs)

    start_finish_times = np.zeros((num_jobs,2))
    start_finish_times[:,1] = -1

    avail_qpus = list(range(n**2//4 * num_ToR))
    arrival_times_iter = arrival_times.copy()
    qpu_reqs_iter = qpu_reqs.copy()
    idx_remain_to_exec = list(range(num_jobs))

    dags_list = {}
    circ_depth_list = {}
    dags_qubit_map = {}
    compute_qs_list = {}
    qpu_assign = {}
    active_jobs = []

    remain_gates = np.array([])
    tic = 0.0
    num_rej = 0
    rejected_idx = []

    # while len(dags_list)< num_jobs or remain_gates.sum()>0:
    while len(dags_list) + num_rej < num_jobs or remain_gates.sum()>0:

        idx_to_exec = np.argwhere(arrival_times_iter<=tic)
        if len(idx_to_exec)>0:
            # print("----------------------")
            # print(f"time: {tic}")
            # print("remain jobs: ", idx_remain_to_exec)
            # print("avail qpus:", avail_qpus)
            idx_to_exec = idx_to_exec[:,0]
            # print("reqs:", arrival_times_iter[idx_to_exec], np.array(qpu_reqs_iter)[idx_to_exec])
            # print("queued:", np.array(idx_remain_to_exec)[idx_to_exec], np.array(qpu_reqs_iter)[idx_to_exec])
            counter = 0
            idx_exec = []
            while len(avail_qpus)>0 and counter<len(idx_to_exec):
                num_req_qpu = qpu_reqs_iter[idx_to_exec[counter]]
                if len(avail_qpus)>= num_req_qpu:
                    idx_exec.append(idx_to_exec[counter])
                    start_finish_times[idx_remain_to_exec[idx_to_exec[counter]],0] = tic
                    start_finish_times[idx_remain_to_exec[idx_to_exec[counter]],1] = 0
                    # start_finish_times[num_jobs-len(arrival_times_iter)+counter,2] = avail_qpus[0]
                    qpu_list = avail_qpus[:num_req_qpu]
                    qpu_assign[idx_remain_to_exec[idx_to_exec[counter]]] = qpu_list
                    compute_qs = [qs_per_node*j + i  for j in qpu_list for i in range(qs_per_node)]
                    num_gates = 10*len(compute_qs)
                    dag, circ_depth, dag_qubit_map = construct_dag(node_qubit_list, compute_qs, num_gates)
                    dags_list[idx_remain_to_exec[idx_to_exec[counter]]] = dag
                    compute_qs_list[idx_remain_to_exec[idx_to_exec[counter]]] = compute_qs
                    circ_depth_list[idx_remain_to_exec[idx_to_exec[counter]]] = circ_depth
                    dags_qubit_map[idx_remain_to_exec[idx_to_exec[counter]]] = dag_qubit_map # don't forget to remove it.
                    for q1 in qpu_list:
                        avail_qpus.remove(q1)
                    active_jobs.append(idx_remain_to_exec[idx_to_exec[counter]])

                counter += 1
            # arrival_times_iter = np.delete(arrival_times_iter, idx_exec)
            # idx_remain_to_exec = np.delete(idx_remain_to_exec, idx_exec)
            #### change to handle buffer size
            idx_remain = sorted( list(set(idx_to_exec) - set(idx_exec)) )
            num_rej += len(idx_remain[buffer:])
            rejected_idx += np.array(idx_remain_to_exec)[idx_remain[buffer:]].tolist()
            idx_remove = idx_remain[buffer:] + idx_exec
            arrival_times_iter = np.delete(arrival_times_iter, idx_remove)
            idx_remain_to_exec = np.delete(idx_remain_to_exec, idx_remove)
            qpu_reqs_iter = np.delete(qpu_reqs_iter, idx_remove)

        num_active_jobs = len(dags_list)
        G_ins =  G.copy()
        num_ir_swap = 0
        num_tel_swap = 0
        execute = True
        while execute:
            execute = False

            indep_gate_seq_list = {}
            dag_node_seq_list = {}
            for i_dag, dag in dags_list.items():
                dag_qubit_map = dags_qubit_map[i_dag]
                compute_qs = compute_qs_list[i_dag]
                indep_gate_seq = []
                dag_node_seq = []
                num_decendants = []
                for node in dag.front_layer():
                    if node.op.num_qubits< 2:
                        dag.remove_op_node(node)
                    if node.op.num_qubits>= 2:
                        indep_gate_seq.append((node_qubit_list[compute_qs[dag_qubit_map[node.qargs[0]]]],node_qubit_list[compute_qs[dag_qubit_map[node.qargs[1]]]]))
                        dag_node_seq.append(node)
                        num_decendants.append(len([g for g in dag.bfs_successors(node)])-1)

                sorted_idx = sorted(range(len(num_decendants)), key=lambda k: num_decendants[k], reverse=True)
                # sorted_idx = sorted(range(len(num_decendants)), key=lambda x: random.random())
                dag_node_seq = [dag_node_seq[k] for k in sorted_idx]
                indep_gate_seq = [indep_gate_seq[k] for k in sorted_idx]
                indep_gate_seq_list[i_dag] = indep_gate_seq
                dag_node_seq_list[i_dag] = dag_node_seq

            rand_idx = sorted(active_jobs, key=lambda x: random.random())
            for i_job in rand_idx:
                indep_gate_seq = indep_gate_seq_list[i_job]
                # for i_g, g in enumerate(indep_gate_seq):
                for i_g, g in enumerate(indep_gate_seq[:1]):
                    n0 = g[0]
                    n1 = g[1]
                    if nx.has_path(G_ins,n0,n1):
                        paths = nx.all_shortest_paths(G_ins, n0, n1, weight=None)
                        for shortestpath in paths:
                            if len(shortestpath)<= 3 :
                                dags_list[i_job].remove_op_node(dag_node_seq_list[i_job][i_g])
                                execute = True
                                break
                            elif len(shortestpath)> 5 :
                                tel_ir = "tel"
                            else:
                                tel_ir = "ir"

                            sp = []
                            b = []
                            # for i in range(0,len(shortestpath)-1):
                            for i in range(1,len(shortestpath)-2):
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
                                
                                dags_list[i_job].remove_op_node(dag_node_seq_list[i_job][i_g])
                                execute = True
                                break
        t_tel = 1/telecom_gen_rate * time_spdc([num_tel_swap])[0]
        t_nir = qubit_reset * time_nir[num_ir_swap]
        dt = max([t_tel,t_nir]) + switch_duration * ( (num_tel_swap + num_ir_swap) > 0 )
        if dt > 0:
            tic += dt
        else:
            if len(arrival_times_iter)>0:
                tic = arrival_times_iter[0]
                
        if len(dags_list) > 0:
            remain_gates = np.ones(num_jobs)
            remain_gates[rejected_idx] = 0
            for i_rem in dags_list.keys():
                remain_gates[i_rem] = len(dags_list[i_rem].gate_nodes())
            # remain_gates = np.array([len(dag.gate_nodes()) for dag in dags_list.values()])
            done_jobs = np.argwhere(remain_gates==0)
            if len(done_jobs)>0:
                done_jobs = done_jobs[:,0]
                for i_jobs in done_jobs:
                    if start_finish_times[i_jobs,1] == 0 and i_jobs not in rejected_idx:
                        # print(f"{i_jobs} is done at {tic}")
                        start_finish_times[i_jobs,1] = tic
                        active_jobs.remove(i_jobs)
                        # avail_qpus += [int(start_finish_times[i_jobs,2])]
                        avail_qpus += qpu_assign[i_jobs]
                avail_qpus = sorted(avail_qpus)

    # compute_time_list = {}
    # for qpu in qpu_vals:
    #     compute_time_list[qpu] = []

    # for i_job in range(num_jobs):
    #     compute_time_list[qpu_reqs[i_job]].append(start_finish_times[i_job,1]-start_finish_times[i_job,0]) 

    # qpu_time = {}
    # for qpu in compute_time_list.keys():
    #     if len(compute_time_list[qpu])>0:
    #         qpu_time[qpu] = sum(compute_time_list[qpu])/len(compute_time_list[qpu])
    execute_time_list = {}
    completion_time_list = {}
    for qpu in qpu_vals:
        execute_time_list[qpu] = []
        completion_time_list[qpu] = []
    for i_job in range(buffer, num_jobs):
        dt = start_finish_times[i_job,1]-start_finish_times[i_job,0]
        # if dt > 0:
        if i_job not in rejected_idx:
            execute_time_list[qpu_reqs[i_job]].append(dt) 
            completion_time_list[qpu_reqs[i_job]].append(start_finish_times[i_job,1]-arrival_times[i_job])   
            
    # print("dt", start_finish_times[:,1]-start_finish_times[:,0])
    # print("exec:", execute_time_list)
    # print("comp:", completion_time_list)
    
    qpu_time = {}
    comp_time = {}
    for qpu in execute_time_list.keys():
        if len(execute_time_list[qpu])>0:
            qpu_time[qpu] = sum(execute_time_list[qpu])/len(execute_time_list[qpu])
            comp_time[qpu] = sum(completion_time_list[qpu])/len(completion_time_list[qpu])

    # print("mean exec src:", qpu_time)
    # print("mean comp src:", comp_time)
    print(f"{num_rej} was rejected out of {num_jobs}: {num_rej/num_jobs}")
    return qpu_time, comp_time, circ_depth_list

def clos_job_scheduler_buffer(specs, G, vertex_list, arrival_times):
    buffer = specs["buffer_size"]
    n = specs["num_sw_ports"]
    telecom_gen_rate = specs["telecom_gen_rate"]
    switch_duration = specs["switch_duration"]
    num_ToR = specs["num_ToR"]
    qs_per_node = specs["qs_per_node"]
    _, _, node_qubit_list = vertex_list

    # telecom_gen_rate = 1/(1e-2) # ebit average generation time in sec
    # switch_duration = 1e-3 # average switching delay in sec

    num_jobs = len(arrival_times)
    start_finish_times = np.zeros((num_jobs,3))
    start_finish_times[:,1] = -1
    idx_remain_to_exec = list(range(num_jobs))

    avail_qpus = list(range(num_ToR))
    arrival_times_iter = arrival_times.copy()

    dags_list = {}
    circ_depth_list = {}
    dags_qubit_map = {}
    compute_qs_list = {}
    active_jobs = []
    qpu_runtime = {}
    for i in range(num_ToR):
        qpu_runtime[i] = []

    remain_gates = np.array([])
    tic = 0.0
    num_rej = 0
    rejected_idx = []
    # while len(dags_list)< num_jobs or remain_gates.sum()>0:
    while len(dags_list) + num_rej < num_jobs or remain_gates.sum()>0:

        idx_to_exec = np.argwhere(arrival_times_iter<=tic)
        if len(idx_to_exec)>0:
            # print(f"time: {tic}")
            # print("avail qpus:", avail_qpus)
            idx_to_exec = idx_to_exec[:,0]
            # print("reqs:", arrival_times_iter[idx_to_exec])
            counter = 0
            idx_exec = []
            while len(avail_qpus)>0 and counter<len(idx_to_exec):
                idx_exec.append(idx_to_exec[counter])
                start_finish_times[idx_remain_to_exec[idx_to_exec[counter]],0] = tic
                start_finish_times[idx_remain_to_exec[idx_to_exec[counter]],1] = 0
                start_finish_times[idx_remain_to_exec[idx_to_exec[counter]],2] = avail_qpus[0]
                compute_qs = [qs_per_node*num_ToR*j + i + qs_per_node*avail_qpus[0]  for j in range(n**2//4) for i in range(qs_per_node)]
                num_gates = 10*len(compute_qs)
                dag, circ_depth, dag_qubit_map = construct_dag(node_qubit_list, compute_qs, num_gates)
                # dags_list.append(dag)
                # compute_qs_list.append(compute_qs)
                # circ_depth_list.append(circ_depth)
                # dags_qubit_map.append(dag_qubit_map) # don't forget to remove it.
                dags_list[idx_remain_to_exec[idx_to_exec[counter]]] = dag
                compute_qs_list[idx_remain_to_exec[idx_to_exec[counter]]] = compute_qs
                circ_depth_list[idx_remain_to_exec[idx_to_exec[counter]]] = circ_depth
                dags_qubit_map[idx_remain_to_exec[idx_to_exec[counter]]] = dag_qubit_map 
                active_jobs.append(idx_remain_to_exec[idx_to_exec[counter]])
                avail_qpus.remove(avail_qpus[0])
                counter += 1

            # arrival_times_iter = np.delete(arrival_times_iter, idx_exec)
            idx_remain = sorted( list(set(idx_to_exec) - set(idx_exec)) )
            num_rej += len(idx_remain[buffer:])
            rejected_idx += np.array(idx_remain_to_exec)[idx_remain[buffer:]].tolist()
            idx_remove = idx_remain[buffer:] + idx_exec
            arrival_times_iter = np.delete(arrival_times_iter, idx_remove)
            idx_remain_to_exec = np.delete(idx_remain_to_exec, idx_remove)

        num_active_jobs = len(dags_list)
        G_ins =  G.copy()
        num_ir_swap = 0
        num_tel_swap = 0
        execute = True
        while execute:
            execute = False

            indep_gate_seq_list = {}
            dag_node_seq_list = {}
            for i_dag, dag in dags_list.items():
                dag_qubit_map = dags_qubit_map[i_dag]
                compute_qs = compute_qs_list[i_dag]
                indep_gate_seq = []
                dag_node_seq = []
                num_decendants = []
                for node in dag.front_layer():
                    if node.op.num_qubits< 2:
                        dag.remove_op_node(node)
                    if node.op.num_qubits>= 2:
                        indep_gate_seq.append((node_qubit_list[compute_qs[dag_qubit_map[node.qargs[0]]]],node_qubit_list[compute_qs[dag_qubit_map[node.qargs[1]]]]))
                        dag_node_seq.append(node)
                        num_decendants.append(len([g for g in dag.bfs_successors(node)])-1)

                sorted_idx = sorted(range(len(num_decendants)), key=lambda k: num_decendants[k], reverse=True)
                # sorted_idx = sorted(range(len(num_decendants)), key=lambda x: random.random())
                dag_node_seq = [dag_node_seq[k] for k in sorted_idx]
                indep_gate_seq = [indep_gate_seq[k] for k in sorted_idx]
                indep_gate_seq_list[i_dag] = indep_gate_seq
                dag_node_seq_list[i_dag] = dag_node_seq

            rand_idx = sorted(active_jobs, key=lambda x: random.random())
            for i_job in rand_idx:
                indep_gate_seq = indep_gate_seq_list[i_job]
                for i_g, g in enumerate(indep_gate_seq):
                    n0 = g[0]
                    n1 = g[1]
                    if nx.has_path(G_ins,n0,n1):
                        paths = nx.all_shortest_paths(G_ins, n0, n1, weight=None)
                        for shortestpath in paths:
                            if len(shortestpath)<= 3 :
                                dags_list[i_job].remove_op_node(dag_node_seq_list[i_job][i_g])
                                execute = True
                                break
                            elif len(shortestpath)> 5 :
                                tel_ir = "tel"
                            else:
                                tel_ir = "ir"

                            sp = []
                            b = []
                            # for i in range(0,len(shortestpath)-1):
                            for i in range(1,len(shortestpath)-2):
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
                                
                                dags_list[i_job].remove_op_node(dag_node_seq_list[i_job][i_g])
                                execute = True
                                break
        dt = 1/telecom_gen_rate * time_spdc([num_tel_swap])[0] + switch_duration * ( num_tel_swap > 0 )
        if dt > 0:
            tic += dt
        else:
            if len(arrival_times_iter)>0:
                tic = arrival_times_iter[0]
                
        if len(dags_list) > 0:
            remain_gates = np.ones(num_jobs)
            remain_gates[rejected_idx] = 0
            for i_rem in dags_list.keys():
                remain_gates[i_rem] = len(dags_list[i_rem].gate_nodes())
            done_jobs = np.argwhere(remain_gates==0)
            if len(done_jobs)>0:
                done_jobs = done_jobs[:,0]
                for i_jobs in done_jobs:
                    if start_finish_times[i_jobs,1] == 0 and i_jobs not in rejected_idx:
                        # print(f"{i_jobs} is done at {tic}")
                        start_finish_times[i_jobs,1] = tic
                        active_jobs.remove(i_jobs)
                        qpu_runtime[int(start_finish_times[i_jobs,2])].append([start_finish_times[i_jobs,0],start_finish_times[i_jobs,1]]) 
                        avail_qpus += [int(start_finish_times[i_jobs,2])]

    # print("dt", start_finish_times[:,1]-start_finish_times[:,0])
    # print(f"{num_rej} was rejected out of {num_jobs}: {num_rej/num_jobs}")

    accepted_idx = sorted(list(set(range(num_jobs))-set(rejected_idx)))
    execute_time_list = []
    completion_time_list = []
    # for i_job in range(buffer+num_ToR, num_jobs):
    for i_job in accepted_idx[num_ToR:]:
        dt = start_finish_times[i_job,1]-start_finish_times[i_job,0]
        # if i_job not in rejected_idx:
        assert dt> 0
        execute_time_list.append(dt) 
        completion_time_list.append(start_finish_times[i_job,1]-arrival_times[i_job])   

    total_time = start_finish_times[:,1].max()
    t_list = np.linspace(0,total_time,int(1e5))
    usage_list = np.zeros(len(t_list))
    for qpu in qpu_runtime.keys():
        for job in qpu_runtime[qpu]:
            t1 = np.argwhere(t_list > job[0])[0,0]
            t2 = np.argwhere(t_list <= job[1])[-1,0]
            usage_list[t1:t2] += 1
            
    qpu_usage = []
    unit_time = t_list[1] - t_list[0]
    for i in range(num_ToR+1):
        qpu_usage.append(len(np.argwhere(usage_list == i))*unit_time/total_time)

    # print("exec:", execute_time_list)
    # print("comp:", completion_time_list)

    time_sum = [[sum(execute_time_list),len(execute_time_list)],[sum(completion_time_list),len(completion_time_list)]]
    rej_vec = [num_jobs, num_rej]
    return time_sum, rej_vec, qpu_usage, circ_depth_list


def clos_job_scheduler(specs, G, vertex_list, arrival_times):
    n = specs["num_sw_ports"]
    telecom_gen_rate = specs["telecom_gen_rate"]
    switch_duration = specs["switch_duration"]
    num_ToR = specs["num_ToR"]
    qs_per_node = specs["qs_per_node"]
    _, _, node_qubit_list = vertex_list

    # telecom_gen_rate = 1/(1e-2) # ebit average generation time in sec
    # switch_duration = 1e-3 # average switching delay in sec

    num_jobs = len(arrival_times)
    start_finish_times = np.zeros((num_jobs,3))
    start_finish_times[:,1] = -1

    avail_qpus = list(range(num_ToR))
    arrival_times_iter = arrival_times.copy()

    dags_list = []
    circ_depth_list = []
    dags_qubit_map = []
    compute_qs_list = []

    remain_gates = np.array([])
    tic = 0.0

    while len(dags_list)< num_jobs or remain_gates.sum()>0:

        idx_to_exec = np.argwhere(arrival_times_iter<=tic)
        if len(idx_to_exec)>0:
            # print(f"time: {tic}")
            # print("avail qpus:", avail_qpus)
            idx_to_exec = idx_to_exec[:,0]
            # print("reqs:", arrival_times_iter[idx_to_exec])
            counter = 0
            idx_exec = []
            while len(avail_qpus)>0 and counter<len(idx_to_exec):
                idx_exec.append(idx_to_exec[counter])
                start_finish_times[num_jobs-len(arrival_times_iter)+counter,0] = tic
                start_finish_times[num_jobs-len(arrival_times_iter)+counter,1] = 0
                start_finish_times[num_jobs-len(arrival_times_iter)+counter,2] = avail_qpus[0]
                compute_qs = [qs_per_node*num_ToR*j + i + qs_per_node*avail_qpus[0]  for j in range(n**2//4) for i in range(qs_per_node)]
                num_gates = 10*len(compute_qs)
                dag, circ_depth, dag_qubit_map = construct_dag(node_qubit_list, compute_qs, num_gates)
                dags_list.append(dag)
                compute_qs_list.append(compute_qs)
                circ_depth_list.append(circ_depth)
                dags_qubit_map.append(dag_qubit_map) # don't forget to remove it.
                avail_qpus.remove(avail_qpus[0])
                counter += 1
            # print("executed:", arrival_times_iter[idx_exec])
            arrival_times_iter = np.delete(arrival_times_iter, idx_exec)

        num_active_jobs = len(dags_list)
        G_ins =  G.copy()
        num_ir_swap = 0
        num_tel_swap = 0
        execute = True
        while execute:
            execute = False

            indep_gate_seq_list = []
            dag_node_seq_list = []
            for i_dag, dag in enumerate(dags_list):
                dag_qubit_map = dags_qubit_map[i_dag]
                compute_qs = compute_qs_list[i_dag]
                indep_gate_seq = []
                dag_node_seq = []
                num_decendants = []
                for node in dag.front_layer():
                    if node.op.num_qubits< 2:
                        dag.remove_op_node(node)
                    if node.op.num_qubits>= 2:
                        indep_gate_seq.append((node_qubit_list[compute_qs[dag_qubit_map[node.qargs[0]]]],node_qubit_list[compute_qs[dag_qubit_map[node.qargs[1]]]]))
                        dag_node_seq.append(node)
                        num_decendants.append(len([g for g in dag.bfs_successors(node)])-1)

                sorted_idx = sorted(range(len(num_decendants)), key=lambda k: num_decendants[k], reverse=True)
                # sorted_idx = sorted(range(len(num_decendants)), key=lambda x: random.random())
                dag_node_seq = [dag_node_seq[k] for k in sorted_idx]
                indep_gate_seq = [indep_gate_seq[k] for k in sorted_idx]
                indep_gate_seq_list.append(indep_gate_seq)
                dag_node_seq_list.append(dag_node_seq)

            rand_idx = sorted(range(num_active_jobs), key=lambda x: random.random())
            for i_job in rand_idx:
                indep_gate_seq = indep_gate_seq_list[i_job]
                for i_g, g in enumerate(indep_gate_seq):
                    n0 = g[0]
                    n1 = g[1]
                    if nx.has_path(G_ins,n0,n1):
                        paths = nx.all_shortest_paths(G_ins, n0, n1, weight=None)
                        for shortestpath in paths:
                            if len(shortestpath)<= 3 :
                                dags_list[i_job].remove_op_node(dag_node_seq_list[i_job][i_g])
                                execute = True
                                break
                            elif len(shortestpath)> 5 :
                                tel_ir = "tel"
                            else:
                                tel_ir = "ir"

                            sp = []
                            b = []
                            # for i in range(0,len(shortestpath)-1):
                            for i in range(1,len(shortestpath)-2):
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
                                
                                dags_list[i_job].remove_op_node(dag_node_seq_list[i_job][i_g])
                                execute = True
                                break
        dt = 1/telecom_gen_rate * time_spdc([num_tel_swap])[0] + switch_duration * ( num_tel_swap > 0 )
        if dt > 0:
            tic += dt
        else:
            if len(arrival_times_iter)>0:
                tic = arrival_times_iter[0]
                
        if len(dags_list) > 0:
            remain_gates = np.array([len(dag.gate_nodes()) for dag in dags_list])
            done_jobs = np.argwhere(remain_gates==0)
            if len(done_jobs)>0:
                done_jobs = done_jobs[:,0]
                for i_jobs in done_jobs:
                    if start_finish_times[i_jobs,1] == 0:
                        # print(f"{i_jobs} is done at {tic}")
                        start_finish_times[i_jobs,1] = tic
                        avail_qpus += [int(start_finish_times[i_jobs,2])]

    return start_finish_times, circ_depth_list

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
            ####
            # g_exec = [] ###
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
                                # ####
                                # g_exec.append(g)
                                # #####
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
            #####
            # print(g_exec)
            # print(switch_time)
            # break
        #########
        switch_seq.append(switch_time)
        # 1/gen_rate*time_spdc(np.array(switch_time)).sum() + switch_duration*len(switch_time)
    return switch_seq

def eff_network_latency_dag_multiqubit_hybrid(G, vertex_list, gate_seq_input):
    edge_switches, node_list, node_qubit_list = vertex_list
    num_nodes = len(node_list)
    num_edge = len(edge_switches)
    num_qubits = len(node_qubit_list)

    qubit_nx_to_qiskit = {qubit: idx for idx, qubit in enumerate(node_qubit_list)}

    qiskit_q_list = QuantumRegister(num_qubits, "q")
    circ = QuantumCircuit(qiskit_q_list)

    for g in gate_seq_input:
        circ.cx(qubit_nx_to_qiskit[g[0]], qubit_nx_to_qiskit[g[1]])

    dag = circuit_to_dag(circ)

    # num_decendants = []
    # for node in dag.front_layer():
    #     num_decendants.append(len([g for g in dag.bfs_successors(node)])-1)
    circ_depth = dag.depth() #max(num_decendants)

    dag_qubit_map = {bit: index for index, bit in enumerate(dag.qubits)}
    switch_seq = []

    while len(dag.gate_nodes())>0:

        G_ins =  G.copy()
        num_ir_swap = 0
        num_tel_swap = 0
        execute = True
        while execute:
            execute = False

            indep_gate_seq = []
            dag_node_seq = []
            num_decendants = []
            for node in dag.front_layer():
                if node.op.num_qubits< 2:
                    dag.remove_op_node(node)
                if node.op.num_qubits>= 2:
                    indep_gate_seq.append((node_qubit_list[dag_qubit_map[node.qargs[0]]],node_qubit_list[dag_qubit_map[node.qargs[1]]]))
                    dag_node_seq.append(node)
                    num_decendants.append(len([g for g in dag.bfs_successors(node)])-1)

            sorted_idx = sorted(range(len(num_decendants)), key=lambda k: num_decendants[k], reverse=True)
            # sorted_idx = sorted(range(len(num_decendants)), key=lambda x: random.random())
            dag_node_seq = [dag_node_seq[k] for k in sorted_idx]
            indep_gate_seq = [indep_gate_seq[k] for k in sorted_idx]

            for i_g, g in enumerate(indep_gate_seq):
                # print(g)
                n0 = g[0]
                n1 = g[1]
                if nx.has_path(G_ins,n0,n1):
                    paths = nx.all_shortest_paths(G_ins, n0, n1, weight=None)
                    for shortestpath in paths:
                        if len(shortestpath)<= 3 :
                            dag.remove_op_node(dag_node_seq[i_g])
                            execute = True
                            break
                        elif len(shortestpath)> 5 :
                            tel_ir = "tel"
                        else:
                            tel_ir = "ir"

                        sp = []
                        b = []
                        # for i in range(0,len(shortestpath)-1):
                        for i in range(1,len(shortestpath)-2):
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
                            
                            dag.remove_op_node(dag_node_seq[i_g])
                            execute = True
                            break

        switch_seq.append([num_ir_swap, num_tel_swap])
    return switch_seq, circ_depth

def network_latency_dag_multiqubit_hybrid(G, vertex_list, gate_seq_input):

    edge_switches, node_list, node_qubit_list = vertex_list
    num_nodes = len(node_list)
    num_edge = len(edge_switches)
    num_qubits = len(node_qubit_list)
    qubit_nx_to_qiskit = {qubit: idx for idx, qubit in enumerate(node_qubit_list)}

    qiskit_q_list = QuantumRegister(num_qubits, "q")
    circ = QuantumCircuit(qiskit_q_list)

    for g in gate_seq_input:
        circ.cx(qubit_nx_to_qiskit[g[0]], qubit_nx_to_qiskit[g[1]])

    dag = circuit_to_dag(circ)
    circ_depth = dag.depth() #max(num_decendants)
    dag_qubit_map = {bit: index for index, bit in enumerate(dag.qubits)}
    switch_seq = []

    while len(dag.gate_nodes())>0:

        indep_gate_seq = []
        dag_node_seq = []
        num_decendants = []
        for node in dag.front_layer():
            if node.op.num_qubits< 2:
                dag.remove_op_node(node)
            if node.op.num_qubits>= 2:
                # gate_set.append((dag_qubit_map[node.qargs[0]],dag_qubit_map[node.qargs[1]]))
                indep_gate_seq.append((node_qubit_list[dag_qubit_map[node.qargs[0]]],node_qubit_list[dag_qubit_map[node.qargs[1]]]))
                dag_node_seq.append(node)
                num_decendants.append(len([g for g in dag.bfs_successors(node)])-1)

        sorted_idx = sorted(range(len(num_decendants)), key=lambda k: num_decendants[k], reverse=True)
        # sorted_idx = sorted(range(len(num_decendants)), key=lambda x: random.random())
        dag_node_seq = [dag_node_seq[k] for k in sorted_idx]
        indep_gate_seq = [indep_gate_seq[k] for k in sorted_idx]

        num_ir_swap = 0
        num_tel_swap = 0
        G_ins =  G.copy()

        for i_g, g in enumerate(indep_gate_seq):
            n0 = g[0]
            n1 = g[1]
            if nx.has_path(G_ins,n0,n1):
                paths = nx.all_shortest_paths(G_ins, n0, n1, weight=None)
                for shortestpath in paths:
                    if len(shortestpath)<= 3 :
                        dag.remove_op_node(dag_node_seq[i_g])
                        break
                    elif len(shortestpath)> 5 :
                        tel_ir = "tel"
                    else:
                        tel_ir = "ir"

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
                        
                        dag.remove_op_node(dag_node_seq[i_g])
                        break

        switch_seq.append([num_ir_swap, num_tel_swap])
        
    return switch_seq, circ_depth

# def parallel_circuit_gen(qubit_list, num_gates):
#     num_nodes = len(node_list)
#     qubit_list = range(qubits_per_node)

#     node_qubit_list = []
#     for node in node_list:
#         for qubit in range(qubits_per_node):
#             node_qubit_list.append((f"{node},{qubit}"))

#     connections = []
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if np.random.rand() > 0.5:
#                 connections.append((node_list[i],node_list[j]))
#             else:
#                 connections.append((node_list[j],node_list[i]))

#     gate_seq_nodes = random.choices(connections, k=num_gates)

#     gate_seq_input = []
#     for n1, n2 in gate_seq_nodes:
#         if n1 == n2:
#             q1 = random.sample(qubit_list,1)[0]
#             q2 = random.sample(list(set(qubit_list)-{q1}),1)[0]
#             gate_seq_input.append((f"{n1},{q1}",f"{n2},{q2}"))
#         else:
#             gate_seq_input.append((f"{n1},{random.sample(qubit_list,1)[0]}",f"{n2},{random.sample(qubit_list,1)[0]}"))


#     return gate_seq

def parallel_query_gen(node_list, qubits_per_node, num_gates):
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
            # connections.append((node_list[i],node_list[j]))
            if np.random.rand() > 0.5:
                connections.append((node_list[i],node_list[j]))
            else:
                connections.append((node_list[j],node_list[i]))

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
    for qpu in node_list:
        attrs[qpu] = {"comm": bandwidth}
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

