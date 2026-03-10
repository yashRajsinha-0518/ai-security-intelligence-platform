from fastapi.testclient import TestClient
import sys
import os

# Ensure the root directory is in the path so we can import 'api'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_validation_error():
    # Missing 34 features out of 35
    data = {"dur": 0.12}
    response = client.post("/predict", json=data)
    # Should throw 422 Unprocessable Entity due to Pydantic schema validation
    assert response.status_code == 422

def test_predict_success():
    # Valid schema payload equivalent to what the dashboard sends
    data = {
        "dur": 0.121478, "proto": "tcp", "service": "ftp", "state": "FIN",
        "spkts": 6.0, "dpkts": 8.0, "sbytes": 258.0, "dbytes": 350.0,
        "rate": 111.954434, "sload": 14158.94238, "dload": 20689.83203, "sloss": 1.0,
        "dloss": 2.0, "sinpkt": 24.295601, "dinpkt": 17.354, "sjit": 14.659851,
        "djit": 19.348616, "swin": 255.0, "stcpb": 2367469771.0, "dtcpb": 988892225.0,
        "dwin": 255.0, "tcprtt": 0.063, "synack": 0.024, "ackdat": 0.039,
        "smean": 43.0, "dmean": 44.0, "trans_depth": 1.0, "response_body_len": 0.0,
        "ct_src_dport_ltm": 1.0, "ct_dst_sport_ltm": 1.0, "is_ftp_login": 1.0,
        "ct_ftp_cmd": 1.0, "ct_flw_http_mthd": 0.0, "is_sm_ips_ports": 0.0,
        "attack_cat": "Normal"
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "label" in response.json()
