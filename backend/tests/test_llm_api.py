from threading import Event
import os
import sys

from s2s_server_pipeline import LanguageModelAPIHandler


def test_llm_api():
    stop_event = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model_name = "deepseek-v3-241226"
    model_url = "https://ark.cn-beijing.volces.com/api/v3"

    lm = LanguageModelAPIHandler(
        stop_event,
        cur_conn_end_event,
        0,
        0,  # placeholder
        interruption_event=interruption_event,
        model_name=model_name,
        model_url=model_url,
        generate_questions=False
    )
    inputs = {
        'data': '如何评价马斯克',
        'user_input_count': 1,
        'uid': 'test_uid',
        'audio_input': False,
    }
    generator = lm.process(inputs)
    outputs = "".join([t["answer_text"] for t in generator])
    print(f'end: {outputs}')


if __name__ == "__main__":
    test_llm_api()
