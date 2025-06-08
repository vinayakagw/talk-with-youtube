import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import numpy as np
import os
import time

DEFAULT_SAMPLE_RATE = 44100  # Standard sample rate for audio
DEFAULT_CHANNELS = 1         # Mono audio

def record_audio(duration: int = 5, sample_rate: int = DEFAULT_SAMPLE_RATE, channels: int = DEFAULT_CHANNELS, output_filename: str = "temp_user_voice.wav", output_path: str = "user_audio") -> str | None:
    """
    Records audio from the default microphone for a specified duration and saves it as a WAV file.

    Args:
        duration: Duration of the recording in seconds.
        sample_rate: The sample rate for the recording.
        channels: Number of audio channels (1 for mono, 2 for stereo).
        output_filename: The name of the output WAV file.
        output_path: The directory to save the audio file.

    Returns:
        The full path to the saved audio file, or None if recording fails.
    """
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
            print(f"Created directory: {output_path}")
        except OSError as e:
            print(f"Error creating directory {output_path}: {e}")
            return None
            
    full_file_path = os.path.join(output_path, output_filename)

    try:
        print(f"Recording for {duration} seconds... Speak now!")
        # Record audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()  # Wait until recording is finished
        
        # Save the recording as a WAV file
        write_wav(full_file_path, sample_rate, recording)
        print(f"Recording saved to {full_file_path}")
        return full_file_path
    except Exception as e:
        print(f"An error occurred during audio recording or saving: {e}")
        # Check for common sounddevice issues
        if "No input device available" in str(e) or "PortAudioError" in str(e):
            print("Error: No input device found or PortAudio error. Please ensure a microphone is connected and configured.")
        return None

if __name__ == '__main__':
    print("Testing audio recording function...")
    
    # Define a path relative to the current file for testing
    test_audio_dir = os.path.join(os.path.dirname(__file__), "..", "test_recordings") # Puts it in project_root/test_recordings
    
    recorded_file = record_audio(duration=5, output_path=test_audio_dir, output_filename="test_recording.wav")
    
    if recorded_file and os.path.exists(recorded_file):
        print(f"Test recording successful. File saved at: {recorded_file}")
        # You can try playing this file with any audio player to verify.
    else:
        print("Test recording failed or file not saved.")
    
    print("\nAvailable audio devices:")
    try:
        print(sd.query_devices())
    except Exception as e:
        print(f"Could not query audio devices: {e}")