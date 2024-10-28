from threading import Event
import os
import sys
import torchaudio

sys.path.append('..')
from s2s_server_pipeline import ParaFormerSTTHandler


def main():
    audio_dir = os.getenv("STT_AUDIO_DIR")
    assert audio_dir is not None
    model_name = os.getenv("STT_MODEL_NAME")
    stop_event = Event()
    should_listen = Event()
    interruption_event = Event()
    cur_conn_end_event = Event()
    model = ParaFormerSTTHandler(
        stop_event,
        cur_conn_end_event,
        0,
        0,  # placeholder
        model_name=model_name,
    )
    # test audio input
    for item in os.listdir(audio_dir):
        if not item.split('.')[-1] in ['wav', 'mp3']:
            continue
        path = os.path.join(audio_dir, item)
        data_wav, sample_rate = torchaudio.load(path)
        data_wav = data_wav.numpy()
        inputs = {
            'data': data_wav[0],  # only the first channel
            'user_input_count': 1,
            'uid': 'test_uid',
        }
        try:
            generator = model.process(inputs)
            outputs = [t for t in generator]
        except:
            continue
        assert all(o['audio_input'] for o in outputs)
        prompt = [o['data'] for o in outputs]
        print(f'{item} end: {prompt}')
    # test audio input
    text_input = 'text input example'
    inputs = {
        'data': text_input,
        'user_input_count': 1,
        'uid': 'test_uid',
    }
    generator = model.process(inputs)
    outputs = [t for t in generator]
    assert outputs[0]['data'] == text_input
    assert not outputs[0]['audio_input']
    print(f'{text_input} end')


if __name__ == "__main__":
    main()
