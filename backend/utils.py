import torchaudio
import librosa
import soundfile as sf


# Function to adjust volume
def adjust_volume(input_path: str, output_path: str, volume_factor: float):
    """
    Adjust the volume of an audio file by a given factor.
    """
    waveform, sample_rate = torchaudio.load(input_path)
    adjusted_waveform = waveform * volume_factor
    adjusted_waveform = adjusted_waveform.clamp(min=-1.0, max=1.0)
    torchaudio.save(output_path, adjusted_waveform, sample_rate)


# Function to change speed
def change_speed(input_path: str, output_path: str, speed: float):
    """
    Change the speed of an audio file by a given percentage.
    """
    # Load the audio file
    audio, sample_rate = librosa.load(input_path)

    # Change the speed
    changed_audio = librosa.effects.time_stretch(audio, rate=speed)

    # Save the modified audio to a new file
    sf.write(output_path, changed_audio, sample_rate)


# Function to shift pitch
def shift_pitch(input_path: str, output_path: str, n_steps: int):
    """
    Shift the pitch of an audio file by a given number of steps.
    """
    y, sr = librosa.load(input_path)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    sf.write(output_path, y_shifted, sr)
