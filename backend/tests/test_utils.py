import torchaudio
import torch
from pydub import AudioSegment
import librosa
import soundfile as sf
import unittest

sys.path.append('..')
from utils import adjust_volume, change_speed, shift_pitch


# Unit tests
class TestAudioProcessing(unittest.TestCase):
    def test_adjust_volume(self):
        input_path = "./ref_audio/ref_wav/ref_audio_default.wav"
        output_path = "./ref_audio/ref_wav/test_adjusted_volume.wav"
        volume_factor = 100
        adjust_volume(input_path, output_path, volume_factor)
        
        # Load the output file and check if it's been modified
        waveform, sample_rate = torchaudio.load(output_path)
        self.assertIsNotNone(waveform)


    def test_change_speed(self):
        input_path = "./ref_audio/ref_wav/ref_audio_default.wav"
        output_path = "./ref_audio/ref_wav/test_speed_changed.wav"
        speed = 50
        change_speed(input_path, output_path, speed)
        
        # Load the output file and check if it's been modified
        result_audio = AudioSegment.from_file(output_path)
        self.assertGreater(result_audio.frame_rate, 0)
        self.assertNotEqual(result_audio.duration_seconds, 0)

    def test_shift_pitch(self):
        input_path = "./ref_audio/ref_wav/ref_audio_default.wav"
        output_path = "./ref_audio/ref_wav/test_pitch_shifted.wav"
        n_steps = 2
        shift_pitch(input_path, output_path, n_steps)
        
        # Load the output file and check if it's been modified
        y_shifted, sr = librosa.load(output_path)
        self.assertIsNotNone(y_shifted)

