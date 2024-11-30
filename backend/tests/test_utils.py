import os
import librosa
import sys

import numpy as np

sys.path.append("..")
from utils import adjust_volume, change_speed, shift_pitch

if __name__ == "__main__":

    def test_adjust_volume():
        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "../ref_audio/ref_wav/test")
        ):
            os.makedirs(
                os.path.join(os.path.dirname(__file__), "../ref_audio/ref_wav/test")
            )
        input_path = os.path.join(
            os.path.dirname(__file__), "../ref_audio/ref_wav/ref_audio_default.wav"
        )

        audio, sample_rate = librosa.load(input_path)
        audio_1 = adjust_volume(audio, volume_factor=0.5)
        audio_2 = adjust_volume(audio, volume_factor=2.0)

        # compute power of the waveform
        power = (audio**2).sum().item()
        power_1 = (audio_1**2).sum().item()
        power_2 = (audio_2**2).sum().item()

        # check if the power has been adjusted
        assert power_1 < power
        assert power_2 > power

    def test_change_speed():
        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "../ref_audio/ref_wav/test")
        ):
            os.makedirs(
                os.path.join(os.path.dirname(__file__), "../ref_audio/ref_wav/test")
            )
        input_path = os.path.join(
            os.path.dirname(__file__), "../ref_audio/ref_wav/ref_audio_default.wav"
        )

        audio, sample_rate = librosa.load(input_path)
        audio_1 = change_speed(audio, speed=0.5)
        audio_2 = change_speed(audio, speed=2.0)

        # check if the length of the audio has been adjusted
        assert len(audio) < len(audio_1)
        assert len(audio_2) < len(audio)

    def test_shift_pitch():
        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "../ref_audio/ref_wav/test")
        ):
            os.makedirs(
                os.path.join(os.path.dirname(__file__), "../ref_audio/ref_wav/test")
            )
        input_path = os.path.join(
            os.path.dirname(__file__), "../ref_audio/ref_wav/ref_audio_default.wav"
        )

        n_steps = [-12, -6, 0, 6, 12]
        audio, sample_rate = librosa.load(input_path)
        for n in n_steps:
            audio_shifted = shift_pitch(audio, sample_rate, n_steps=n)
            assert audio_shifted is not None

    test_adjust_volume()
    test_change_speed()
    test_shift_pitch()
