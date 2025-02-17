from threading import Event
import os
import sys

sys.path.append('..')
from audio_server_pipeline import ASRHandler


def main():
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

    file_path = "res/只用一个文.mp3"

    response = model.process(file_path)
    print(response)

if __name__ == "__main__":
    main()
