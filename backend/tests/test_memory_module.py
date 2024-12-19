from threading import Event
import os
import sys

from dotenv import load_dotenv
load_dotenv(".dev.env")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from s2s_server_pipeline_memory import LanguageModelAPIHandlerWithMemory


def main():
    stop_event = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model_name = "deepseek-chat"
    model_url = "https://api.deepseek.com"
    embedding_model_name = os.getenv("EMBEDDING_MODEL_PATH")

    lm = LanguageModelAPIHandlerWithMemory(
        stop_event,
        cur_conn_end_event,
        0,
        0,  # placeholder
        mode=1,
        interruption_event=interruption_event,
        model_name=model_name,
        model_url=model_url,
        generate_questions=False,
    )
    inputs = {
        'data': '如何评价马斯克',
        'user_input_count': 1,
        'uid': 'test_uid',
        'audio_input': False,
    }
    generator = lm.process(inputs)
    print([t["answer_text"] for t in generator])
    outputs = "".join([t["answer_text"] for t in generator])
    print(f'end: {outputs}')


if __name__ == "__main__":
    main()
