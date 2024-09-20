import numpy as np
import time
from joblib import Parallel, delayed
import multiprocessing
num_cores = 28                                 


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
    return event_times

hom_coincidence = lambda x, dw: np.exp(-(x*dw) **2/2 ) 

gen_rate = 1e6 # Parameter lambda in Hz
total_time = 1e0  # Total time in sec for the process
# linewidth_list = np.array([1,3,10])*1e9 # photon linewidth in GHz
linewidth_list = np.array([3])*1e9 # photon linewidth in GHz

# reset_time_list = np.array([0.1,1,3,10])*1e-6
reset_time_list = np.array([3])*1e-6

Niter = 1000
Nrep = 20 # No. of repetitions

for i_l, dw in enumerate(linewidth_list):
    for i_reset, reset_time in enumerate(reset_time_list):
        def runner(i_rep):
            tic = time.time()
            succ_time = []
            for _ in range(Niter):
            # print(i_r, end="\r")
                s1 = poisson_random_process(gen_rate, total_time)
                s2 = poisson_random_process(gen_rate, total_time)

                num_trials = min(s1.shape[0],s2.shape[0])
                # num_trials
                all_events = np.concatenate((s1,s2))
                events_inds = all_events.argsort()
                all_events = all_events[events_inds]
                emissions = np.zeros(events_inds.shape[0], dtype=np.int32)
                emissions[np.argwhere(events_inds>=s1.shape[0])] = 1

                # time_diff = []

                qu_reset = [False, False]
                qu_avaialble_time = [0,0]
                i_trial = 0
                while i_trial < len(emissions)-1:
                    if all_events[i_trial] >= qu_avaialble_time[emissions[i_trial]]:
                        qu_reset[emissions[i_trial]] = False
                        qu_avaialble_time[emissions[i_trial]] = 0
                    
                    if emissions[i_trial+1]== 1- emissions[i_trial]:
                        if all_events[i_trial+1] >= qu_avaialble_time[emissions[i_trial+1]]:
                            qu_reset[emissions[i_trial+1]] = False
                            qu_avaialble_time[emissions[i_trial+1]] = 0

                        if (not qu_reset[0])*(not qu_reset[1]):
                            dt = all_events[i_trial+1]-all_events[i_trial]
                            prob = 0.5* hom_coincidence(dt,dw) # 0.5 for BSM post selection
                            if np.random.rand()< prob: # accept with probability p
                        #     ## check if BSM can be performed
                                ## BSM measurement
                                succ_time.append(all_events[i_trial+1])
                                break
                            # i_trial += 2
                            
                    # ## shift timer
                    # new_avail_time = all_events[i_trial]+reset_time
                    if  all_events[i_trial] >= qu_avaialble_time[emissions[i_trial]]:
                        qu_reset[emissions[i_trial]] = True
                        qu_avaialble_time[emissions[i_trial]] = all_events[i_trial]+reset_time
                        
                    i_trial += 1


            succ_time = np.array(succ_time)
            fname = f"results/dist/lw_{dw/1e9:.2f}_qres_{reset_time*1e6:.1f}_r_{i_rep}.npz"
            toc = time.time()
            print(f"{fname}, events: {succ_time.shape}, elapsed time {toc-tic} sec")
            np.savez(fname, gen_rate, total_time, succ_time)
        
        results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(Nrep))

print("Finished!")