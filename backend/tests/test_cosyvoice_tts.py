from threading import Event
import os
import sys
import torchaudio

sys.path.append('..')
from s2s_server_pipeline import CosyVoiceTTSHandler


def main():
    ref_dir = os.getenv("TTS_REF_DIR")
    model_name = os.getenv("TTS_MODEL_NAME")
    stop_event = Event()
    should_listen = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model = CosyVoiceTTSHandler(
        stop_event,
        cur_conn_end_event,
        0,
        0,  # placeholder
        should_listen=should_listen,
        interruption_event=interruption_event,
        ref_dir=ref_dir,
        model_name=model_name,
    )
    inputs = {
        'question_text': '如何评价马斯克',
        'answer_text': '我不知道如何评价，我是一个专业助手而不是一个锐评家。',
        'user_input_count': 1,
        'end_flag': False,
    }
    generator = model.process(inputs, return_np=False)
    outputs = [t["answer_audio"] for t in generator]
    torchaudio.save('cosyvoice_output.wav', outputs[0], 16000)

    print(f'end: {outputs}')


if __name__ == "__main__":
    main()
