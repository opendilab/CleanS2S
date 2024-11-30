from threading import Event
import os
import sys

sys.path.append('..')
from s2s_server_pipeline_rag import RAGLanguageModelHelper, RAGLanguageModelAPIHandler


def check_environ():
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not embedding_model_name:
        assert False, 'No embedding model name given.'
    llm_api_key = os.getenv("LLM_API_KEY")
    if not llm_api_key:
        assert False, 'No llm api key given.'


def main():
    stop_event = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model_name = "deepseek-chat"
    model_url = "https://api.deepseek.com"
    try:
        check_environ()
    except AssertionError:
        print('Something is wrong with your environment variables, please check it.')
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
    outputs = ''
    for t in generator:
        if isinstance(t, str):
            outputs += t
        elif isinstance(t, dict):
            outputs += ''.join(list(t.values()))
    print(f'end: {outputs}')


if __name__ == "__main__":
    main()
