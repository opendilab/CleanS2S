from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
import uuid
from threading import Event
from typing import Optional
import logging
import threading
from fastapi.testclient import TestClient
import random

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from s2s_server_pipeline import LanguageModelAPIHandler
from uid import UidManager

app = FastAPI()

stop_event = Event()
interruption_event = Event()
cur_conn_end_event = Event()
uid_manager = UidManager(max_uid_count=5, uid_timeout_second=1)

# model config
model_name = "deepseek-v3-241226"
model_url = "https://ark.cn-beijing.volces.com/api/v3"

user_lms = {}

lm = LanguageModelAPIHandler(
    stop_event,
    cur_conn_end_event,
    0,
    0,  # placeholder
    interruption_event=interruption_event,
    model_name=model_name,
    model_url=model_url,
    generate_questions=False,
    uid_manager=uid_manager
)

class RequestBody(BaseModel):
    user_input: str
    uid: Optional[str] = None  # uid is optional

@app.post("/process")
def process_input(request_body: RequestBody):
    user_input = request_body.user_input
    uid = request_body.uid or str(uuid.uuid4())  # if uid is not provided, generate a new uid

    inputs = {
        'data': user_input,
        'user_input_count': 1,
        'uid': uid,
        'audio_input': False
    }

    generator = lm.process(inputs)

    content = [t["answer_text"] for t in generator]
    outputs = "".join(content)

    return {"outputs": outputs, "uid": uid} 


client = TestClient(app)

uid_list = []
for i in range(5):
    new_uid = str(uuid.uuid4())
    uid_list.append(new_uid)

def test_process_with_uid():
    """request with uid"""

    id = random.randint(0,4)
    test_uid = uid_list[id]

    response = client.post(
        "/process",
        json={"user_input": "Hello, how are you?", "uid": test_uid},
    )
    assert response.status_code == 200
    data = response.json()
    assert "outputs" in data
    assert "uid" in data
    assert data["uid"] == test_uid

def test_process_without_uid():
    """request without uid"""
    response = client.post(
        "/process",
        json={"user_input": "Hello, how are you?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "outputs" in data
    assert "uid" in data
    assert isinstance(data["uid"], str)  


if __name__ == "__main__":
    threads = []
    for _ in range(10):
        t1 = threading.Thread(target=test_process_with_uid)
        t2 = threading.Thread(target=test_process_without_uid)
        threads.extend([t1, t2])
        t1.start()
        t2.start()
    
    for t in threads:
        t.join()
