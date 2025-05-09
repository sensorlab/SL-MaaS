import requests
from typing import List, Tuple
import requests
import pickle
import numpy as np

# Data
data_url = '../nvg_inference_data.pkl'
# labels = ["Normal", "SlowD", "SuddenD", "SuddenR", "InstaD"] 

with open(data_url, 'rb') as f:
    data_all = pickle.load(f)

data_t = np.expand_dims(data_all[0], axis=0)

import time

def test_nvg_endpoint():
    # Wait for the container to be ready
    time.sleep(3)

    response = requests.post("http://localhost:8000/NVG", json=data_t.tolist())
    assert response.status_code == 200