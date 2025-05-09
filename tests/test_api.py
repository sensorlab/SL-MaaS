import requests

def test_root():
    r = requests.get("http://localhost:8000/NVG")
    assert r.status_code == 200


"""import requests
import numpy as np
import time

def test_nvg_endpoint():
    # Wait for the container to be ready
    time.sleep(3)

    # Create dummy input: 300 values flattened into a list of list of floats
    input_data = np.random.rand(300).reshape(100, 3).tolist()

    response = requests.post("http://localhost:8000/NVG", json=input_data)
    assert response.status_code == 200

    result = response.json()
    assert "pred" in result
    assert result["pred"].isdigit() or result["pred"].startswith("tensor")"""