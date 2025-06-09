import os
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

# Initialize conversation history in session if needed
@app.before_request
def initialize_session():
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    if 'chat_mode' not in session:
        session['chat_mode'] = 'text'  # Default to text mode


@app.route('/', methods=['GET', 'POST'])
async def index():
    """
    Main page route - handles YouTube URL submission and displays the interface
    """
    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url', '').strip()
        if not youtube_url:
            flash('Please enter a YouTube URL', 'error')
            return redirect(url_for('index'))
            
        try:
            # Reset any previous conversation when processing a new video
            session['conversation_history'] = []
            
            # Download the YouTube video's audio
            audio_file = await download_youtube_audio(
                youtube_url=youtube_url,
                output_dir=AUDIO_OUTPUT_DIR
            )
            
            if not audio_file:
                flash('Failed to download audio from the YouTube video', 'error')
                return redirect(url_for('index'))
                
            # Transcribe the audio file
            transcript = await transcribe_audio_sarvam_batch(audio_file)
            
            if not transcript:
                flash('Failed to transcribe the video audio', 'error')
                return redirect(url_for('index'))
                
            # Store the transcript and URL in the session
            session['current_transcript'] = transcript
            session['current_youtube_url'] = youtube_url
            
            flash('Video processed successfully!', 'success')
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"Error processing YouTube URL: {e}")
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('index'))

    # For GET requests: render the main page with current data
    return render_template('index.html', 
                           transcript=session.get('current_transcript'), 
                           youtube_url=session.get('current_youtube_url'))


@app.route('/record_and_transcribe', methods=['POST'])
async def record_and_transcribe_route():
    """
    API endpoint that receives audio recorded from the user's browser and transcribes it.
    """
    if 'audio' not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file received."
        }), 400
        
    audio_file = request.files['audio']
    
    # Get requested duration (now sent as form data)
    recording_duration = int(request.form.get('duration', 30))
    
    timestamp = int(time.time())
    temp_recording_filename = f"user_voice_{timestamp}.wav"
    temp_recording_path = os.path.join(USER_RECORDINGS_DIR, temp_recording_filename)
    
    # Create directory if it doesn't exist
    if not os.path.exists(USER_RECORDINGS_DIR):
        os.makedirs(USER_RECORDINGS_DIR)
    
    # Save the uploaded audio file
    audio_file.save(temp_recording_path)
    
    if not os.path.exists(temp_recording_path):
        return jsonify({
            "success": False,
            "error": "Failed to save audio file."
        }), 500

    # Check if API key is available for transcription
    if not os.getenv("SARVAM_API_KEY"):
        return jsonify({
            "success": False,
            "error": "API key not configured for transcription."
        }), 500
        
    # Transcribe the recorded audio
    transcript = transcribe_audio_direct(temp_recording_path, language="en")

    # Clean up the recording file
    try:
        os.remove(temp_recording_path)
    except Exception as e:
        logger.warning(f"Could not delete temporary recording {temp_recording_path}: {e}")

    if transcript:
        return jsonify({
            "success": True,
            "transcript": transcript
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to transcribe audio. Please try again."
        }), 500


@app.route('/ask_chat', methods=['POST'])
async def ask_chat():
    """
    API endpoint that answers questions about the transcript in a conversational manner
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
    
    # Get conversation mode (text or voice)
    conversation_mode = data.get('mode', 'text')
    session['chat_mode'] = conversation_mode
    
    # Retrieve conversation history
    conversation_history = session.get('conversation_history', [])
    
    # Add the user's question to history
    conversation_history.append({"role": "user", "content": question})
    
    # Get answer from the AI model with conversation context
    answer_text = get_answer_from_transcript_sarvam(
        transcript, 
        question, 
        conversation_history=conversation_history
    )

    if answer_text:
        logger.info(f"Got answer from AI")
        
        # Add AI's response to conversation history
        conversation_history.append({"role": "assistant", "content": answer_text})
        
        # Keep history at a reasonable size (last 10 messages)
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
            
        # Update session
        session['conversation_history'] = conversation_history
        
        # Check if we can generate speech for the answer
        if not os.getenv("SARVAM_API_KEY"):
             logger.warning("API key not configured for TTS.")
             return jsonify({
                "success": True, 
                "answer_text": answer_text, 
                "answer_audio_url": None,
                "conversation_history": conversation_history,
                "message": "Text-to-speech service not configured."
            })

        # Generate speech for the answer
        timestamp = int(time.time())
        tts_filename = f"chat_answer_{timestamp}.wav"
        
        # Adjust TTS for conversational mode
        voice_style = "conversational" if conversation_mode == "voice" else "neutral"
        
        generated_audio_file_path = text_to_speech_sarvam(
            text_to_speak=answer_text,
            output_audio_path=TTS_OUTPUT_DIR, 
            output_filename=tts_filename,
            language_code="en-IN",  # Indian English
            voice_style=voice_style  # Use the conversational style for voice mode
        )

        # Return both text and audio URL (if available)
        if generated_audio_file_path:
            relative_audio_path = os.path.join("tts_audio", tts_filename)
            audio_url = url_for('static', filename=relative_audio_path)
            logger.info(f"Generated speech audio for answer")
            return jsonify({
                "success": True, 
                "answer_text": answer_text, 
                "answer_audio_url": audio_url,
                "conversation_history": conversation_history
            })
        else:
            logger.error("Failed to generate speech for answer.")
            return jsonify({
                "success": True,  # Still return the text answer
                "answer_text": answer_text, 
                "answer_audio_url": None,
                "conversation_history": conversation_history,
                "message": "Failed to generate audio for the answer."
            })
    else:
        logger.warning("AI model failed to provide an answer.")
        return jsonify({"success": False, "error": "Sorry, I couldn't get an answer from the AI."}), 500


# This code runs when you execute this file directly
if __name__ == '__main__':
    # Ensure directories exist
    if not os.path.exists(AUDIO_OUTPUT_DIR): 
        os.makedirs(AUDIO_OUTPUT_DIR)
    if not os.path.exists(USER_RECORDINGS_DIR): 
        os.makedirs(USER_RECORDINGS_DIR)
    if not os.path.exists(TTS_OUTPUT_DIR): 
        os.makedirs(TTS_OUTPUT_DIR)
    
    # Start the web server
    app.run(debug=True, port=5001)
