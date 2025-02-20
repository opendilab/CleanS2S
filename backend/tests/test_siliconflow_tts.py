from threading import Event
import os
import sys

sys.path.append('..')
from audio_server_pipeline import TTSHandler


def test_tts():
    stop_event = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model = TTSHandler(
        stop_event,
        cur_conn_end_event,
        0,
        0,  # placeholder
        interruption_event=interruption_event,
        ref_dir="ref_audio",
    )
    inputs = "只用一个文件实现的流式全双工语音交互原型智能体！"
    ref_voice = "FunAudioLLM/CosyVoice2-0.5B:alex"
    save_path = "res"

    model.process(inputs, ref_voice, save_path)


if __name__ == "__main__":
    test_tts()
