import os
import time
import logging

# Try to import sounddevice, but don't fail if not available
try:
    import sounddevice as sd
    from scipy.io.wavfile import write as write_wav
    AUDIO_RECORDING_AVAILABLE = True
except (ImportError, OSError):
    AUDIO_RECORDING_AVAILABLE = False
    logging.warning("Audio recording libraries not available in this environment")

# Audio quality settings
DEFAULT_SAMPLE_RATE = 44100  # CD-quality audio (44,100 samples per second)
DEFAULT_CHANNELS = 1         # Mono recording (1 channel instead of stereo's 2)
MAX_RECORDING_SECONDS = 60   # Maximum recording time (to prevent huge files)

# Create a stub/mock version of the recording function
def record_audio(duration: int = MAX_RECORDING_SECONDS, sample_rate: int = DEFAULT_SAMPLE_RATE, 
                channels: int = DEFAULT_CHANNELS, output_filename: str = "temp_user_voice.wav", 
                output_path: str = "user_audio") -> str | None:
    """
    This is a stub function for Vercel deployment.
    The actual recording happens in the browser.
    """
    if not AUDIO_RECORDING_AVAILABLE:
        logging.warning("Server-side audio recording not available")
        return None
    
    logging.warning("Server-side audio recording not available in this environment")
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