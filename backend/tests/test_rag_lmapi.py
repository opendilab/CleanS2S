from threading import Event
import os
import sys

sys.path.append('..')
from s2s_server_pipeline_rag import RAGLanguageModelHelper, RAGLanguageModelAPIHandler


def main():
    stop_event = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model_name = "deepseek-chat"
    model_url = "https://api.deepseek.com"
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")

    rag = RAGLanguageModelHelper(model_name, model_url, 256, embedding_model_name, rag_backend='base')
    # To use LightRAG as rag backend:
    # rag = RAGLanguageModelHelper(model_name, model_url, 256, embedding_model_name, rag_backend='light_rag')

    lm = RAGLanguageModelAPIHandler(
        stop_event,
        cur_conn_end_event,
        0,
        0,  # placeholder
        interruption_event=interruption_event,
        model_name=model_name,
        model_url=model_url,
        rag=rag
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
    main()
