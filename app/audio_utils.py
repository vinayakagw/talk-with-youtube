import sounddevice as sd  # Library for recording audio from microphone
from scipy.io.wavfile import write as write_wav  # For saving audio as WAV files
import numpy as np
import os
import time
import queue
import threading

# Audio quality settings
DEFAULT_SAMPLE_RATE = 44100  # CD-quality audio (44,100 samples per second)
DEFAULT_CHANNELS = 1         # Mono recording (1 channel instead of stereo's 2)
MAX_RECORDING_SECONDS = 60   # Maximum recording time (to prevent huge files)

def record_audio(duration: int = MAX_RECORDING_SECONDS, sample_rate: int = DEFAULT_SAMPLE_RATE, 
                channels: int = DEFAULT_CHANNELS, output_filename: str = "temp_user_voice.wav", 
                output_path: str = "user_audio") -> str | None:
    """
    Records audio from your microphone and saves it as a WAV file.
    
    Args:
        duration: Maximum seconds to record (default: 60)
        sample_rate: Audio quality (higher = better quality but larger file)
        channels: 1 for mono (voice), 2 for stereo
        output_filename: Name of the saved file
        output_path: Folder to save the file in
    
    Returns:
        Path to the saved audio file, or None if recording failed
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
            print(f"Created directory: {output_path}")
        except OSError as e:
            print(f"Error creating directory {output_path}: {e}")
            return None
            
    full_file_path = os.path.join(output_path, output_filename)

    try:
        print(f"Recording for up to {duration} seconds... Speak now!")
        
        # Start recording audio from the microphone
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()  # Wait until recording is finished
        
        # Save the recording as a WAV file
        write_wav(full_file_path, sample_rate, recording)
        print(f"Recording saved to {full_file_path}")
        return full_file_path
        
    except Exception as e:
        print(f"An error occurred during audio recording or saving: {e}")
        # Help diagnose common microphone problems
        if "No input device available" in str(e) or "PortAudioError" in str(e):
            print("Error: No microphone found or it's not accessible. Please check your microphone connection.")
        return None

# This code runs when you execute this file directly (for testing)
if __name__ == '__main__':
    print("Testing audio recording function...")
    
    # Save test recordings in project_root/test_recordings
    test_audio_dir = os.path.join(os.path.dirname(__file__), "..", "test_recordings")
    
    # Try making a 5-second test recording
    recorded_file = record_audio(duration=5, output_path=test_audio_dir, output_filename="test_recording.wav")
    
    if recorded_file and os.path.exists(recorded_file):
        print(f"Test recording successful. File saved at: {recorded_file}")
    else:
        print("Test recording failed or file not saved.")
    
    # List available audio devices to help with troubleshooting
    print("\nAvailable audio devices:")
    try:
        print(sd.query_devices())
    except Exception as e:
        print(f"Could not query audio devices: {e}")