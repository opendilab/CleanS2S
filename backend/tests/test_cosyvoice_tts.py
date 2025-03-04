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
    inputs = [
        {
            'question_text': '如何评价马斯克',
            'answer_text': '我不知道如何评价，我是一个专业助手而不是一个锐评家。',
            'user_input_count': 1,
            'end_flag': False,
            'uid': 'test_uid',
        },
        {
            'question_text': '你需要我做什么',
            'answer_text': '请你复制，我现在正在说的这句话。',
            'user_input_count': 1,
            'end_flag': False,
            'uid': 'test_uid',
        },
        {
            'question_text': '怎么评价',
            'answer_text': '这句对话有点复杂。',
            'user_input_count': 1,
            'end_flag': False,
            'uid': 'test_uid',
        },
        {
            'question_text': '你在纠结什么呢',
            'answer_text': '今天吃什么？这是一个问题。这是一个让我纠结了很久的问题。',
            'user_input_count': 1,
            'end_flag': False,
            'uid': 'test_uid',
        },
        {
            'question_text': '给我讲个故事吧',
            'answer_text': '这是一个很复杂的故事，故事的起源来源于一个古老的传说。',
            'user_input_count': 1,
            'end_flag': False,
            'uid': 'test_uid',
        },
        {
            'question_text': '你听到了吗',
            'answer_text': '你说啥？',
            'user_input_count': 1,
            'end_flag': False,
            'uid': 'test_uid',
        },
        {
            'question_text': '你为什么这么开心',
            'answer_text': '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
            'user_input_count': 1,
            'end_flag': False,
            'uid': 'test_uid',
        },
    ]
    for i, item in enumerate(inputs):
        generator = model.process(item, return_np=False)
        outputs = [t["answer_audio"] for t in generator]
        torchaudio.save(f'cosyvoice_output_{i}.wav', outputs[0], 16000)
        print(f'item {i} end: {outputs}')


if __name__ == "__main__":
    main()
