from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
import uuid
from threading import Event
from typing import Optional
import logging

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
