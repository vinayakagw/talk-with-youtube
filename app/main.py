import os
import asyncio
import time
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from app.youtube_utils import download_youtube_audio
from app.sarvam_ai import (
    transcribe_audio_sarvam_batch, 
    get_answer_from_transcript_sarvam,
    transcribe_audio_direct,
    text_to_speech_sarvam
)
from app.audio_utils import record_audio
from dotenv import load_dotenv

# Set up logging to track what's happening in the app
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Find the project root directory and load environment variables from .env file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

# Create the Flask web application
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")  # Required for session and flash messages

# Set up directories for file storage
AUDIO_OUTPUT_DIR = os.path.join(project_root, "temp_audio")  # For YouTube video audio
USER_RECORDINGS_DIR = os.path.join(project_root, "temp_user_recordings")  # For user's voice recordings
TTS_OUTPUT_DIR = os.path.join(app.static_folder, "tts_audio")  # For AI-generated speech (web-accessible)

# Create these directories if they don't exist
if not os.path.exists(USER_RECORDINGS_DIR):
    os.makedirs(USER_RECORDINGS_DIR)
if not os.path.exists(TTS_OUTPUT_DIR):
    os.makedirs(TTS_OUTPUT_DIR)


@app.route('/', methods=['GET', 'POST'])
async def index():
    """
    Main page route - handles YouTube URL submission and displays the interface
    """
    if request.method == 'POST':
        # User submitted a YouTube URL
        youtube_url = request.form.get('youtube_url')
        if not youtube_url:
            flash("Please enter a YouTube video URL.", "error")
            return redirect(url_for('index'))

        # Store URL in session and show processing message
        session['current_youtube_url'] = youtube_url
        logger.info(f"Processing YouTube URL: {youtube_url}")
        flash(f"Processing YouTube URL: {youtube_url}. This may take a moment...", "info")

        # Reset previous state for new video
        session['current_transcript'] = None 
        session.pop('current_question', None)
        session.pop('current_answer', None)
        session.pop('current_answer_audio_url', None)

        # Step 1: Download the audio from the YouTube video
        downloaded_audio_path = download_youtube_audio(youtube_url, output_path=AUDIO_OUTPUT_DIR)

        if downloaded_audio_path:
            # Step 2: Transcribe the audio (this may take a while)
            flash("Audio downloaded successfully. Starting transcription (this may take some time)...", "info")
            logger.info(f"Audio downloaded, starting transcription.")
            transcript_result = await transcribe_audio_sarvam_batch(downloaded_audio_path) 

            if transcript_result:
                # Store transcript in session and show success message
                session['current_transcript'] = transcript_result
                flash("Transcription successful! You can now chat with the tutor.", "success")
                logger.info("Transcription successful.")
            else:
                flash("Could not transcribe the audio. Check server logs for details.", "error")
                logger.error("Transcription failed.")
            
            # Optionally: Clean up downloaded audio file to save space
            # try:
            #     if os.path.exists(downloaded_audio_path): os.remove(downloaded_audio_path)
            # except Exception as e:
            #     logger.error(f"Error cleaning up audio file: {e}")
        else:
            flash("Failed to download audio. Please check the URL and try again.", "error")
            logger.error(f"Failed to download audio for URL: {youtube_url}")
        
        return redirect(url_for('index'))

    # For GET requests: render the main page with current data
    return render_template('index.html', 
                           transcript=session.get('current_transcript'), 
                           youtube_url=session.get('current_youtube_url'))


@app.route('/record_and_transcribe', methods=['POST'])
async def record_and_transcribe_route():
    """
    API endpoint that records audio from the user's microphone and transcribes it.
    Supports variable recording durations up to MAX_RECORDING_SECONDS.
    """
    # Get requested duration (default to MAX_RECORDING_SECONDS if not specified)
    data = request.get_json() or {}
    recording_duration = min(int(data.get('duration', 30)), 60)  # Cap at 60 seconds max
    
    timestamp = int(time.time())
    temp_recording_filename = f"user_voice_{timestamp}.wav"
    
    recorded_audio_path = record_audio(
        duration=recording_duration,
        output_path=USER_RECORDINGS_DIR,
        output_filename=temp_recording_filename
    )

    if not recorded_audio_path:
        return jsonify({"success": False, "error": "Failed to record audio."}), 500

    # Check if API key is available for transcription
    if not os.getenv("SARVAM_API_KEY"):
        logger.error("SARVAM_API_KEY not configured.")
        # Clean up the recording file
        try:
            if os.path.exists(recorded_audio_path): os.remove(recorded_audio_path)
        except Exception as e:
            logger.error(f"Error cleaning up recording: {e}")
        return jsonify({"success": False, "error": "Speech-to-text service not configured."}), 500
        
    # Transcribe the recorded audio
    transcript = transcribe_audio_direct(recorded_audio_path, language="en")

    # Clean up the recording file
    try:
        if os.path.exists(recorded_audio_path):
            os.remove(recorded_audio_path)
    except Exception as e:
        logger.error(f"Error cleaning up recording: {e}")

    # Return the transcribed text as JSON
    if transcript:
        return jsonify({"success": True, "transcript": transcript})
    else:
        return jsonify({"success": False, "error": "Failed to transcribe audio."}), 500


@app.route('/ask_chat', methods=['POST'])
async def ask_chat():
    """
    API endpoint that answers questions about the transcript
    """
    # Get the transcript from session
    transcript = session.get('current_transcript')
    if not transcript:
        logger.warning("No transcript available.")
        return jsonify({"success": False, "error": "No transcript available. Please process a video first."}), 400

    # Get the question from the request
    data = request.get_json()
    if not data or 'question' not in data:
        logger.warning("No question provided.")
        return jsonify({"success": False, "error": "No question provided."}), 400
    
    question = data['question']
    logger.info(f"Received question: {question}")

    # Get answer from the AI model
    answer_text = get_answer_from_transcript_sarvam(transcript, question)

    if answer_text:
        logger.info(f"Got answer from AI")
        
        # Check if we can generate speech for the answer
        if not os.getenv("SARVAM_API_KEY"):
             logger.warning("API key not configured for TTS.")
             return jsonify({
                "success": True, 
                "answer_text": answer_text, 
                "answer_audio_url": None,
                "message": "Text-to-speech service not configured."
            })

        # Generate speech for the answer
        timestamp = int(time.time())
        tts_filename = f"chat_answer_{timestamp}.wav"
        
        generated_audio_file_path = text_to_speech_sarvam(
            text_to_speak=answer_text,
            output_audio_path=TTS_OUTPUT_DIR, 
            output_filename=tts_filename,
            language_code="en-IN"  # Indian English
        )

        # Return both text and audio URL (if available)
        if generated_audio_file_path:
            relative_audio_path = os.path.join("tts_audio", tts_filename)
            audio_url = url_for('static', filename=relative_audio_path)
            logger.info(f"Generated speech audio for answer")
            return jsonify({
                "success": True, 
                "answer_text": answer_text, 
                "answer_audio_url": audio_url
            })
        else:
            logger.error("Failed to generate speech for answer.")
            return jsonify({
                "success": True,  # Still return the text answer
                "answer_text": answer_text, 
                "answer_audio_url": None,
                "message": "Failed to generate audio for the answer."
            })
    else:
        logger.warning("AI model failed to provide an answer.")
        return jsonify({"success": False, "error": "Sorry, I couldn't get an answer from the AI."}), 500


# This code runs when you execute this file directly
if __name__ == '__main__':
    # Ensure directories exist
    if not os.path.exists(AUDIO_OUTPUT_DIR): os.makedirs(AUDIO_OUTPUT_DIR)
    if not os.path.exists(USER_RECORDINGS_DIR): os.makedirs(USER_RECORDINGS_DIR)
    if not os.path.exists(TTS_OUTPUT_DIR): os.makedirs(TTS_OUTPUT_DIR)
    
    # Start the web server
    app.run(debug=True, port=5001)
