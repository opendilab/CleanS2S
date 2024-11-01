import librosa
import numpy as np


# Function to adjust volume
def adjust_volume(audio, volume_factor: float):
    """
    Adjust the volume of an audio file by a given factor.
    """
    adjusted_audio = audio * volume_factor
    adjusted_audio = np.clip(adjusted_audio, -1.0, 1.0)
    return adjusted_audio


# Function to change speed
def change_speed(audio, speed: float):
    """
    Change the speed of an audio file by a given percentage.
    """
    changed_audio = librosa.effects.time_stretch(audio, rate=speed)
    return changed_audio


# Function to shift pitch
def shift_pitch(audio, sampling_rate, n_steps: int):
    """
    Shift the pitch of an audio file by a given number of steps.
    """
    audio_shifted = librosa.effects.pitch_shift(
        audio, sr=sampling_rate, n_steps=n_steps
    )
    return audio_shifted
