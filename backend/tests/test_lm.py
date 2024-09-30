from threading import Event
import os
import sys

sys.path.append('..')
from s2s_server_pipeline import LanguageModelHandler


def main():
    stop_event = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model_name = os.getenv("LLM_MODEL_NAME")
    assert model_name is not None, "Please indicate local LLM model name, like `/home/root/Qwen=7B`"

    lm = LanguageModelHandler(
        stop_event,
        cur_conn_end_event,
        0,
        0,  # placeholder
        interruption_event=interruption_event,
        model_name=model_name,
    )
    inputs = {
        'data': '如何评价马斯克',
        'user_input_count': 1,
    }
    generator = lm.process(inputs)
    outputs = "".join([t["answer_text"] for t in generator])
    print(f'end: {outputs}')


if __name__ == "__main__":
    main()
