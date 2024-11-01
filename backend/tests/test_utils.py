import os
import librosa
import soundfile as sf
import unittest
import sys

sys.path.append('..')
from utils import adjust_volume, change_speed, shift_pitch

# Unit tests
class TestAudioProcessing(unittest.TestCase):
    def test_adjust_volume(self):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test')):
            os.makedirs(os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test'))
        input_path = os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/ref_audio_default.wav')
        output_path_1 = os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test/adjusted_volume_1.wav')
        output_path_2 = os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test/adjusted_volume_2.wav')

        adjust_volume(input_path, output_path_1, volume_factor = 0.5)
        adjust_volume(input_path, output_path_2, volume_factor = 2.0)
        
        # Load the output file and check if it's been modified
        waveform, sr = librosa.load(input_path)
        waveform_1, sr = librosa.load(output_path_1)
        waveform_2, sr = librosa.load(output_path_2)
        self.assertIsNotNone(waveform)
        self.assertIsNotNone(waveform_1)
        self.assertIsNotNone(waveform_2)

        # compute power of the waveform
        power = waveform.pow(2).sum().item()
        power_1 = waveform_1.pow(2).sum().item()
        power_2 = waveform_2.pow(2).sum().item()

        # check if the power has been adjusted
        self.assertGreater(power, power_1)
        self.assertGreater(power_2, power)

    def test_change_speed(self):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test')):
            os.makedirs(os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test'))
        input_path = os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/ref_audio_default.wav')
        output_path_1 = os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test/speed_changed_1.wav')
        output_path_2 = os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test/speed_changed_2.wav')

        change_speed(input_path, output_path_1, speed = 0.5)
        change_speed(input_path, output_path_2, speed = 2.0)
        
        # Load the output file and check if it's been modified
        y, sr = librosa.load(input_path)
        y_1, sr = librosa.load(output_path_1)
        y_2, sr = librosa.load(output_path_2)
        self.assertIsNotNone(y)
        self.assertIsNotNone(y_1)
        self.assertIsNotNone(y_2)
        # check if the length of the audio has been adjusted
        self.assertGreater(len(y_1), len(y))
        self.assertGreater(len(y), len(y_2))
    

    def test_shift_pitch(self):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test')):
            os.makedirs(os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test'))
        input_path = os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/ref_audio_default.wav')
        output_path = os.path.join(os.path.dirname(__file__), '../ref_audio/ref_wav/test/pitch_shifted.wav')
        n_steps = 2
        shift_pitch(input_path, output_path, n_steps)
        
        # Load the output file and check if it's been modified
        y_shifted, sr = librosa.load(output_path)
        self.assertIsNotNone(y_shifted)
