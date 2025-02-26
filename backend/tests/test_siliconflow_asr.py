from threading import Event
import os
import sys

from s2s_server_pipeline import ASRHandler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def test_asr():
    stop_event = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model = ASRHandler(
        stop_event,
        cur_conn_end_event,
        0,
        0,  # placeholder
        interruption_event=interruption_event,
    )

    file_path = os.path.join(PROJECT_ROOT, "backend/ref_audio/ref_wav/ref_audio_2.wav")

    response = model.process(file_path)
    assert isinstance(response, str), "response type is wrong"
    print(response)

if __name__ == "__main__":
    test_asr()
