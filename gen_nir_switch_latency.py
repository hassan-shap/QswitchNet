from network_utils_hybrid import *
import random
import json

nir_prob = 1e-2 # NIR gen prob
switch_time_list = np.arange(1,41)
nir_latency_list = time_nir(switch_time_list, nir_prob).tolist()
nir_latency_list = [0] + nir_latency_list

JSON_PATH = "data/nir_latency.json"
with open(JSON_PATH, 'w') as json_file:
    json_file.write(json.dumps(nir_latency_list) + '\n')

print(nir_latency_list)
