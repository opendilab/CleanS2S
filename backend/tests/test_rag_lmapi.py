from threading import Event
import os
import sys

sys.path.append('..')
from s2s_server_pipeline_rag import RAGLanguageModelHelper, RAGLanguageModelAPIHandler
from s2s_server_pipeline import logger


def check_illegal_environ():
    """
    Check if the environment variables are set.
    """
    environs = ['EMBEDDING_MODEL_NAME', 'LLM_API_KEY']
    illegals = []
    for env in environs:
        if os.getenv(env) is None:
            illegals.append(env)
    return illegals


def main():
    stop_event = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model_name = "deepseek-chat"
    model_url = "https://api.deepseek.com"
    bad_vars = check_illegal_environ()
    if len(bad_vars) > 0:
        logger.info(f"Some environment variables are not set, which can be problematic: {bad_vars}")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")

    # Use traditional RAG as rag backend:
    # rag = RAGLanguageModelHelper(model_name, model_url, 256, embedding_model_name, rag_backend='base')

    # Use LightRAG as rag backend:
    rag = RAGLanguageModelHelper(model_name, model_url, 256, embedding_model_name, rag_backend='light_rag')

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
        answer = t['answer_text']
        if isinstance(answer, str):
            outputs += answer
        elif isinstance(answer, dict):
            outputs += ''.join(list(answer.values()))
    print(f'end: {outputs}')


if __name__ == "__main__":
    main()
