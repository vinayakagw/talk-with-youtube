import os
import asyncio
import time # For generating unique filenames
import logging # Import the logging module
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session # Added jsonify and session
from app.youtube_utils import download_youtube_audio
from app.sarvam_ai import (
    transcribe_audio_sarvam_batch, 
    get_answer_from_transcript_sarvam,
    transcribe_audio_direct, # Added
    text_to_speech_sarvam    # Added
)
from app.audio_utils import record_audio # Added
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file located in the project root
# __file__ is app/main.py, so os.path.dirname(__file__) is app/
# os.path.join(os.path.dirname(__file__), '..') goes up one level to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Define where to save the downloaded audio
# We'll save it in a directory called 'temp_audio' in the project root
AUDIO_OUTPUT_DIR = os.path.join(project_root, "temp_audio")
USER_RECORDINGS_DIR = os.path.join(project_root, "temp_user_recordings")
TTS_OUTPUT_DIR = os.path.join(app.static_folder, "tts_audio") # Inside static folder for web access

# Ensure these directories exist
if not os.path.exists(USER_RECORDINGS_DIR):
    os.makedirs(USER_RECORDINGS_DIR)
if not os.path.exists(TTS_OUTPUT_DIR):
    os.makedirs(TTS_OUTPUT_DIR)

# Global variables to store transcript, URL, question, and answer
current_transcript = None
current_youtube_url = None
current_question = None
current_answer = None
current_answer_audio_url = None # New global to store path to TTS audio for the answer


@app.route('/', methods=['GET', 'POST'])
async def index():
    # global current_transcript, current_youtube_url, current_question, current_answer # No longer using these globals for Q&A display
    # global current_answer_audio_url
    
    if request.method == 'POST':
        # This POST is for submitting the YouTube URL
        youtube_url = request.form.get('youtube_url')
        if not youtube_url:
            flash("Please enter a YouTube video URL.", "error")
            return redirect(url_for('index'))

        session['current_youtube_url'] = youtube_url # Keep this for display in the form
        logger.info(f"Processing YouTube URL: {youtube_url}")
        flash(f"Processing YouTube URL: {youtube_url}. This may take a moment...", "info")

        # Reset transcript for a new video
        session['current_transcript'] = None 
        # Q&A state (question, answer, audio_url) is now managed client-side by the chat
        session.pop('current_question', None)
        session.pop('current_answer', None)
        session.pop('current_answer_audio_url', None)


        downloaded_audio_path = download_youtube_audio(youtube_url, output_path=AUDIO_OUTPUT_DIR)

        if downloaded_audio_path:
            flash("Audio downloaded successfully. Starting batch transcription (this may take some time)...", "info")
            logger.info(f"Audio downloaded to: {downloaded_audio_path}, starting transcription.")
            transcript_result = await transcribe_audio_sarvam_batch(downloaded_audio_path) 

            if transcript_result:
                session['current_transcript'] = transcript_result
                flash("Batch transcription successful! You can now chat with the tutor.", "success")
                logger.info("Batch transcription successful.")
            else:
                flash("Could not transcribe the audio using batch process. Check server logs for details.", "error")
                logger.error("Batch transcription failed.")
            
            # Optional: Clean up downloaded YouTube audio file
            # try:
            #     if os.path.exists(downloaded_audio_path): os.remove(downloaded_audio_path)
            # except Exception as e:
            #     logger.error(f"Error cleaning up YouTube audio {downloaded_audio_path}: {e}")
        else:
            flash("Failed to download audio. Please check the URL and try again.", "error")
            logger.error(f"Failed to download audio for URL: {youtube_url}")
        
        return redirect(url_for('index'))

    # For GET request, or after POST processing for URL
    # We only need to pass transcript and youtube_url for the main page structure
    return render_template('index.html', 
                           transcript=session.get('current_transcript'), 
                           youtube_url=session.get('current_youtube_url'))
                           # question, answer, answer_audio_url are removed

@app.route('/record_and_transcribe', methods=['POST'])
async def record_and_transcribe_route():
    """
    Records audio from the user, transcribes it, and returns the transcript.
    """
    recording_duration = 5 # seconds, make configurable if needed
    timestamp = int(time.time())
    temp_recording_filename = f"user_voice_{timestamp}.wav"
    
    recorded_audio_path = record_audio(
        duration=recording_duration,
        output_path=USER_RECORDINGS_DIR,
        output_filename=temp_recording_filename
    )

    if not recorded_audio_path:
        return jsonify({"success": False, "error": "Failed to record audio."}), 500

    # Ensure SARVAM_API_KEY is available for direct STT
    # The STT URL is now hardcoded in sarvam_ai.py, so this check for the URL itself is less critical
    # but we keep the API key check.
    if not os.getenv("SARVAM_API_KEY"): # Check for API Key
        logger.error("SARVAM_API_KEY not configured for direct STT.")
        # Clean up temp recording
        try:
            if os.path.exists(recorded_audio_path): os.remove(recorded_audio_path)
        except Exception as e:
            logger.error(f"Error cleaning up temp recording {recorded_audio_path}: {e}")
        return jsonify({"success": False, "error": "STT service not configured."}), 500
        
    transcript = transcribe_audio_direct(recorded_audio_path, language="en") # Assuming English

    # Clean up the temporary recording file
    try:
        if os.path.exists(recorded_audio_path):
            os.remove(recorded_audio_path)
    except Exception as e:
        logger.error(f"Error cleaning up temporary recording {recorded_audio_path}: {e}")

    if transcript:
        return jsonify({"success": True, "transcript": transcript})
    else:
        return jsonify({"success": False, "error": "Failed to transcribe audio."}), 500

@app.route('/ask_chat', methods=['POST'])
async def ask_chat():
    transcript = session.get('current_transcript')
    if not transcript:
        logger.warning("ask_chat called without transcript in session.")
        return jsonify({"success": False, "error": "No transcript available. Please process a video first."}), 400

    data = request.get_json()
    if not data or 'question' not in data:
        logger.warning("ask_chat called without question in payload.")
        return jsonify({"success": False, "error": "No question provided."}), 400
    
    question = data['question']
    logger.info(f"Received chat question: {question}")

    answer_text = get_answer_from_transcript_sarvam(transcript, question)

    if answer_text:
        logger.info(f"LLM Answer: {answer_text}")
        # Generate TTS for the answer
        # However, if we want to be explicit about configuration:
        if not os.getenv("SARVAM_API_KEY"): # Check for API Key
             logger.warning("SARVAM_API_KEY not configured. Cannot generate TTS for chat.")
             return jsonify({
                "success": True, 
                "answer_text": answer_text, 
                "answer_audio_url": None,
                "message": "TTS service API Key not configured."
            })

        timestamp = int(time.time())
        tts_filename = f"chat_answer_{timestamp}.wav"
        
        generated_audio_file_path = text_to_speech_sarvam(
            text_to_speak=answer_text,
            output_audio_path=TTS_OUTPUT_DIR, 
            output_filename=tts_filename,
            language_code="en-IN" # Adjust as needed
        )

        if generated_audio_file_path:
            relative_audio_path = os.path.join("tts_audio", tts_filename)
            audio_url = url_for('static', filename=relative_audio_path)
            logger.info(f"TTS generated for chat: {audio_url}")
            return jsonify({
                "success": True, 
                "answer_text": answer_text, 
                "answer_audio_url": audio_url
            })
        else:
            logger.error("Failed to generate TTS for chat answer.")
            return jsonify({
                "success": True, # Still success as we got text answer
                "answer_text": answer_text, 
                "answer_audio_url": None,
                "message": "Failed to generate audio for the answer."
            })
    else:
        logger.warning("LLM failed to provide an answer for chat question.")
        return jsonify({"success": False, "error": "Sorry, I couldn't get an answer from the LLM."}), 500

# ... (rest of your Flask app, e.g., if __name__ == '__main__': app.run(...))
if __name__ == '__main__':
    # Ensure directories exist (also done at module level, but good for __main__)
    if not os.path.exists(AUDIO_OUTPUT_DIR): os.makedirs(AUDIO_OUTPUT_DIR)
    if not os.path.exists(USER_RECORDINGS_DIR): os.makedirs(USER_RECORDINGS_DIR)
    if not os.path.exists(TTS_OUTPUT_DIR): os.makedirs(TTS_OUTPUT_DIR)
    
    app.run(debug=True, port=5001)
