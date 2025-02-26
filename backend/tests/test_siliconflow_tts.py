from threading import Event
import os
import sys

from s2s_server_pipeline import TTSHandler

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
    text = "只用一个文件实现的流式全双工语音交互原型智能体！"
    ref_voice = "FunAudioLLM/CosyVoice2-0.5B:alex"
    save_path = "res"

    inputs = {"text": text, "ref_voice": ref_voice, "save_path": save_path}

    audio_np = model.process(inputs)

    print(audio_np.shape)

    if audio_np is not None:
        assert len(audio_np) > 32000


if __name__ == "__main__":
    test_tts()
