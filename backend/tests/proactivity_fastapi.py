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

from s2s_server_pipeline_proactivity import LanguageModelAPIHandlerProactivity, ChatMode

app = FastAPI()

stop_event = Event()
interruption_event = Event()
cur_conn_end_event = Event()

# model config
# model_name = "deepseek-chat"
# model_url = "https://api.deepseek.com"

model_name = "llama-3.3-70B-instruct"
model_url = "http://14.103.16.79:11000/v1"

user_lms = {}

def get_or_create_lm(uid: str):
    if uid not in user_lms:
        lm = LanguageModelAPIHandlerProactivity(
            stop_event,
            cur_conn_end_event,
            0,
            0,  # placeholder
            character='zhangwei.txt',
            mode=ChatMode.REGULAR_MODE,
            interruption_event=interruption_event,
            model_name=model_name,
            model_url=model_url,
            generate_questions=False,
        )
        user_lms[uid] = lm
    return user_lms[uid]

class RequestBody(BaseModel):
    user_input: str
    uid: Optional[str] = None  # uid is optional

@app.post("/process")
def process_input(request_body: RequestBody):
    user_input = request_body.user_input
    uid = request_body.uid or str(uuid.uuid4())  # if uid is not provided, generate a new uid

    lm = get_or_create_lm(uid)

    inputs = {
        'data': user_input,
        'user_input_count': 1,
        'uid': uid,
        'audio_input': False
    }

    generator = lm.process(inputs)

    content = [t["answer_text"] for t in generator]
    outputs = "".join(content)

    # logging.info(f"USER_LMS: {user_lms}")

    return {"outputs": outputs, "uid": uid} 
