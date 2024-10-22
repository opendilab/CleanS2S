import torchaudio
import torch
from pydub import AudioSegment
import librosa
import soundfile as sf
import unittest

# Function to adjust volume
def adjust_volume(input_path: str, output_path: str, volume_factor: int):
    """
    Adjust the volume of an audio file by a given factor.
    """
    volume_factor = volume_factor / 100.0
    waveform, sample_rate = torchaudio.load(input_path)
    adjusted_waveform = waveform * volume_factor
    adjusted_waveform = adjusted_waveform.clamp(min=-1.0, max=1.0)
    torchaudio.save(output_path, adjusted_waveform, sample_rate)

# Function to change speed
def change_speed(input_path: str, output_path: str, speed: int):
    """
    Change the speed of an audio file by a given percentage.
    """
    speed = 1.0 + speed / 100.0
    audio = AudioSegment.from_file(input_path)
    new_audio = audio.speedup(playback_speed=speed)
    new_audio.export(output_path, format="wav")

# Function to shift pitch
def shift_pitch(input_path: str, output_path: str, n_steps: int):
    """
    Shift the pitch of an audio file by a given number of steps.
    """
    y, sr = librosa.load(input_path)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    sf.write(output_path, y_shifted, sr)
