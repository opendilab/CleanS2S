from threading import Event
import os
import sys

from dotenv import load_dotenv
load_dotenv(".dev.env")

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from s2s_server_pipeline_proactivity import LanguageModelAPIHandlerProactivity, ChatMode, ChatMode


def main():
    stop_event = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model_name = "deepseek-chat"
    model_url = "https://api.deepseek.com"
    embedding_model_name = os.getenv("EMBEDDING_MODEL_PATH")

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
    user_input = ''
    while True:
        user_input = input('输入 exit 退出')
        if user_input == 'exit':
            break
        inputs = {
            'data': user_input,
            'user_input_count': 1,
            'uid': 'test_uid',
            'audio_input': False,
        }
        generator = lm.process(inputs)
        content = [t["answer_text"] for t in generator]
        print(content)
        outputs = "".join(content)
        print(f'end: {outputs}')


if __name__ == "__main__":
    main()
