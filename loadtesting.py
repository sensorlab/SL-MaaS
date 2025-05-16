from typing import List, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy as np

#added a commet to test something

# Functions
def make_request(url: str, data: List) -> Tuple[float, any]:
    response = requests.post(url, json=data)
    rtt = response.elapsed.total_seconds()

    if response.status_code != 200:
        print(f'Error: {response.status_code}')

    if not response.json()['pred']:
        print('Error: Response is empty')

    return rtt, response.json()["pred"]


def test_api_parallel(url: str, num_requests: int, data: List) -> Tuple[List[float], List[any]]:
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        latencies_responses = list(executor.map(lambda _: make_request(url, data), range(num_requests)))

    latencies = [latency for latency, _ in latencies_responses]
    responses = [response for _, response in latencies_responses]
    return latencies, responses


# Data
data_url = 'nvg_inference_data.pkl'
# labels = ["Normal", "SlowD", "SuddenD", "SuddenR", "InstaD"] 

with open(data_url, 'rb') as f:
    data_all = pickle.load(f)

ip = 'localhost'
endpoint_route = "NVG"
port = 8000

url = f"http://{ip}:{port}/{endpoint_route}" # change the port if running on kubernetes

parallel_requests = 100

# For larger amount of parallel requests set open files limit to: ulimit -n 500000

# warmup
print(f'Running warmup for /{endpoint_route} with {parallel_requests} requests')
data_t = np.expand_dims(data_all[0], axis=0)
_, _ = test_api_parallel(url, parallel_requests, data_t.tolist())
print('Done')
