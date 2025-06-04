import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.app.main import app

client = TestClient(app)

def test_upload_endpoint():
    response = client.post('/upload', files={'image': ('test.png', b'x', 'image/png'),
                                             'annotation': ('test.json', b'{}', 'application/json')})
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_train_endpoint():
    response = client.post('/train', json={'epochs': 1, 'batch_size': 1, 'task_type': 'DocVQA'})
    assert response.status_code == 200
    assert 'pid' in response.json()
