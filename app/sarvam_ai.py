import asyncio  # For running operations in parallel without blocking
import aiofiles  # For asynchronous file reading/writing
import requests  # For making HTTP API calls
import json
import os
import time
import mimetypes  # For determining file types
import logging
from urllib.parse import urlparse
from azure.storage.filedatalake.aio import DataLakeDirectoryClient, FileSystemClient  # For Azure storage
from azure.storage.filedatalake import ContentSettings
from dotenv import load_dotenv  # For loading API keys from .env file
from pprint import pprint
import base64  # For encoding/decoding data
import subprocess  # For running external programs like FFmpeg
import shutil  # For file operations like move and delete

# --- Setup and Configuration ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

# API key for Sarvam AI services
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# API URLs for different Sarvam AI services
# Batch Speech-to-Text (for long audio files)
SARVAM_JOB_INIT_URL = "https://api.sarvam.ai/speech-to-text/job/init"  # Initialize a transcription job
SARVAM_JOB_START_URL = "https://api.sarvam.ai/speech-to-text/job"  # Start the job
SARVAM_JOB_STATUS_URL_TEMPLATE = "https://api.sarvam.ai/speech-to-text/job/{job_id}/status"  # Check job status

# Other Sarvam AI services
SARVAM_LLM_API_URL = "https://api.sarvam.ai/v1/chat/completions"  # AI chat model for answering questions
SARVAM_LLM_MODEL_NAME = "sarvam-m"  # The specific AI model to use
SARVAM_REALTIME_STT_API_URL = "https://api.sarvam.ai/speech-to-text"  # For short audio clips
SARVAM_TTS_API_URL = "https://api.sarvam.ai/text-to-speech"  # For converting text to spoken audio

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Azure Storage Client for Batch Transcription ---
class SarvamStorageClient:
    """Handles file uploads/downloads to/from Azure storage for Sarvam batch processing"""
    
    def __init__(self, storage_url_with_sas: str):
        """Set up the client with a storage URL that includes access permissions"""
        self.account_url, self.file_system_name, self.directory_name, self.sas_token = (
            self._extract_url_components(storage_url_with_sas)
        )
        self.lock = asyncio.Lock()
        logger.info(f"Initialized SarvamStorageClient with directory: {self.directory_name}")

    def update_storage_url(self, storage_url_with_sas: str):
        """Update the storage location if needed"""
        self.account_url, self.file_system_name, self.directory_name, self.sas_token = (
            self._extract_url_components(storage_url_with_sas)
        )
        logger.info(f"Updated SarvamStorageClient URL to directory: {self.directory_name}")

    def _extract_url_components(self, url: str):
        """Break down a storage URL into its components"""
        parsed_url = urlparse(url)
        account_url = f"{parsed_url.scheme}://{parsed_url.netloc}".replace(
            ".blob.", ".dfs."  # Azure Data Lake uses .dfs instead of .blob
        )
        path_components = parsed_url.path.strip("/").split("/")
        file_system_name = path_components[0]  # Like a container in Azure
        directory_name = "/".join(path_components[1:])  # Folder path within container
        sas_token = parsed_url.query  # Access token
        return account_url, file_system_name, directory_name, sas_token

    async def upload_file(self, local_file_path: str, overwrite: bool = True) -> bool:
        """Upload a file to Azure storage"""
        logger.info(f"Starting upload of file: {local_file_path}")
        async with DataLakeDirectoryClient(
            account_url=f"{self.account_url}?{self.sas_token}",
            file_system_name=self.file_system_name,
            directory_name=self.directory_name,
            credential=None,  # SAS token is already in the URL
        ) as directory_client:
            file_name = os.path.basename(local_file_path)
            try:
                async with aiofiles.open(local_file_path, mode="rb") as file_data:
                    # Determine file type
                    mime_type, _ = mimetypes.guess_type(local_file_path)
                    if mime_type is None:
                        if local_file_path.lower().endswith(".mp3"):
                            mime_type = "audio/mpeg"
                        else:
                            mime_type = "application/octet-stream"  # Default type
                    
                    file_client = directory_client.get_file_client(file_name)
                    data_bytes = await file_data.read()
                    await file_client.upload_data(
                        data_bytes,
                        overwrite=overwrite,
                        content_settings=ContentSettings(content_type=mime_type),
                    )
                    logger.info(f"✅ File uploaded successfully: {file_name} (Type: {mime_type})")
                    return True
            except Exception as e:
                logger.error(f"❌ Upload failed for {file_name}: {str(e)}")
                return False

    async def list_files(self) -> list[str]:
        """Get a list of files in the Azure storage directory"""
        logger.info(f"\n📂 Listing files in directory: {self.directory_name}...")
        file_names = []
        async with FileSystemClient(
            account_url=f"{self.account_url}?{self.sas_token}",
            file_system_name=self.file_system_name,
            credential=None,
        ) as file_system_client:
            async for path in file_system_client.get_paths(self.directory_name):
                # Extract just the filename from the full path
                if self.directory_name and path.name.startswith(self.directory_name + "/"):
                    file_name = path.name[len(self.directory_name)+1:]
                else:
                    file_name = path.name
                
                if not path.is_directory:  # Skip folders, only include files
                    async with self.lock:
                        file_names.append(file_name)
        logger.info(f"Found {len(file_names)} files: {file_names}")
        return file_names

    async def download_file(self, file_name: str, destination_local_path: str) -> bool:
        """Download a file from Azure storage"""
        logger.info(f"\n⬇️ Starting download of {file_name} to {destination_local_path}")
        # Create destination folder if it doesn't exist
        os.makedirs(os.path.dirname(destination_local_path), exist_ok=True)

        async with DataLakeDirectoryClient(
            account_url=f"{self.account_url}?{self.sas_token}",
            file_system_name=self.file_system_name,
            directory_name=self.directory_name,
            credential=None,
        ) as directory_client:
            try:
                file_client = directory_client.get_file_client(file_name)
                async with aiofiles.open(destination_local_path, mode="wb") as local_file:
                    stream = await file_client.download_file()
                    data = await stream.readall()
                    await local_file.write(data)
                logger.info(f"✅ Downloaded: {file_name} -> {destination_local_path}")
                return True
            except Exception as e:
                logger.error(f"❌ Download failed for {file_name}: {str(e)}")
                return False


# --- Batch Transcription Job Management ---
def _get_headers():
    """Create HTTP headers with the API key"""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not found in environment variables.")
    return {
        "API-Subscription-Key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

def initialize_stt_job():
    """Start a new batch transcription job with Sarvam AI"""
    logger.info("🚀 Initializing STT job...")
    headers = _get_headers()
    # Some APIs don't need Content-Type header
    if "Content-Type" in headers and SARVAM_JOB_INIT_URL == "https://api.sarvam.ai/speech-to-text/job/init":
         init_headers = {"API-Subscription-Key": SARVAM_API_KEY}

    response = requests.post(SARVAM_JOB_INIT_URL, headers=init_headers)
    logger.info(f"Initialize Job Response Status: {response.status_code}")
    if response.status_code == 202:  # Accepted
        job_info = response.json()
        logger.info("Job initialized successfully:")
        pprint(job_info)
        return job_info
    else:
        logger.error(f"Job initialization failed: {response.status_code} - {response.text}")
        return None

def start_stt_job(job_id: str, language_code: str = "unknown"):
    """Begin processing a batch transcription job"""
    logger.info(f"▶️ Starting job: {job_id} with language: {language_code}")
    headers = _get_headers()
    data = {
        "job_id": job_id,
        "job_parameters": {"language_code": language_code}
    }
    logger.info("Request Body for starting job:")
    pprint(data)
    response = requests.post(SARVAM_JOB_START_URL, headers=headers, data=json.dumps(data))
    logger.info(f"Start Job Response Status: {response.status_code}")
    if response.status_code == 200:  # OK
        job_start_info = response.json()
        logger.info("Job started successfully:")
        pprint(job_start_info)
        return job_start_info
    else:
        logger.error(f"Failed to start job: {response.status_code} - {response.text}")
        return None

def check_stt_job_status(job_id: str):
    """Check if a batch transcription job is complete"""
    logger.info(f"🔍 Checking status for job: {job_id}")
    url = SARVAM_JOB_STATUS_URL_TEMPLATE.format(job_id=job_id)
    # GET request typically doesn't need Content-Type in headers
    status_headers = {"API-Subscription-Key": SARVAM_API_KEY}
    response = requests.get(url, headers=status_headers)
    logger.info(f"Job Status Response Status: {response.status_code}")
    if response.status_code == 200:  # OK
        job_status = response.json()
        logger.info("Job status:")
        pprint(job_status)
        return job_status
    else:
        logger.error(f"Failed to get job status: {response.status_code} - {response.text}")
        return None

async def transcribe_audio_sarvam_batch(audio_file_path: str) -> str | None:
    """
    Transcribe a long audio file using Sarvam AI's batch processing service.
    This handles the entire process: upload, transcription, and result download.
    
    Returns the complete transcript text, or None if transcription failed.
    """
    if not SARVAM_API_KEY:
        logger.error("Error: SARVAM_API_KEY not found in environment variables.")
        return None
    if not os.path.exists(audio_file_path):
        logger.error(f"Error: Audio file not found at {audio_file_path}")
        return None

    # Step 1: Initialize a new transcription job
    job_info = initialize_stt_job()
    if not job_info or "job_id" not in job_info:
        return None
    
    job_id = job_info["job_id"]
    input_storage_path_sas = job_info.get("input_storage_path")
    output_storage_path_sas = job_info.get("output_storage_path")

    if not input_storage_path_sas or not output_storage_path_sas:
        logger.error("Missing storage paths in job initialization response.")
        return None

    # Step 2: Upload the audio file to Azure storage
    input_client = SarvamStorageClient(input_storage_path_sas)
    upload_successful = await input_client.upload_file(audio_file_path)
    if not upload_successful:
        logger.error(f"Failed to upload {audio_file_path} to Sarvam storage.")
        return None

    # Step 3: Start the transcription job
    start_response = start_stt_job(job_id, language_code="unknown")
    if not start_response:
        logger.error(f"Failed to start STT job {job_id}.")
        return None

    # Step 4: Wait for the job to complete (poll status)
    logger.info(f"⏳ Monitoring job status for {job_id}...")
    max_retries = 30
    retry_delay_seconds = 20
    
    final_transcript = None

    for attempt in range(max_retries):
        await asyncio.sleep(retry_delay_seconds)
        logger.info(f"Status check attempt {attempt + 1}/{max_retries} for job {job_id}")
        job_status_response = check_stt_job_status(job_id)

        if not job_status_response:
            logger.warning("Failed to get job status on this attempt.")
            continue

        status = job_status_response.get("job_state")
        logger.info(f"Current job state: {status}")

        if status == "Completed":
            logger.info("✅ Job completed successfully!")
            
            # Step 5: Download the transcript result
            output_client = SarvamStorageClient(output_storage_path_sas)
            
            # Find the transcript file among output files
            output_files = await output_client.list_files()
            if not output_files:
                logger.error("No files found in the output storage.")
                return None

            # Look for a JSON or TXT file containing the transcript
            transcript_file_name = None
            for fname in output_files:
                if fname.lower().endswith(".json"):
                    transcript_file_name = fname
                    break
            if not transcript_file_name:  # Try .txt if no .json found
                 for fname in output_files:
                    if fname.lower().endswith(".txt"):
                        transcript_file_name = fname
                        break
            
            # If still no match, take any non-log file or the first file as fallback
            if not transcript_file_name and output_files:
                potential_files = [f for f in output_files if not f.lower().endswith(".log")]
                if potential_files:
                    transcript_file_name = potential_files[0]
                else:
                    transcript_file_name = output_files[0]

            if transcript_file_name:
                # Download and read the transcript file
                temp_download_dir = os.path.join(project_root, "temp_transcripts")
                os.makedirs(temp_download_dir, exist_ok=True)
                local_transcript_path = os.path.join(temp_download_dir, transcript_file_name)
                
                download_ok = await output_client.download_file(transcript_file_name, local_transcript_path)
                if download_ok and os.path.exists(local_transcript_path):
                    logger.info(f"Transcript file downloaded to: {local_transcript_path}")
                    try:
                        # Read the file content
                        async with aiofiles.open(local_transcript_path, "r", encoding="utf-8") as f:
                            content = await f.read()
                        
                        # Extract the transcript text based on file format
                        if transcript_file_name.lower().endswith(".json"):
                            try:
                                transcript_json = json.loads(content)
                                # Try various common keys for transcript text
                                final_transcript = transcript_json.get("transcript") or \
                                                   transcript_json.get("text") or \
                                                   transcript_json.get("full_transcript") 
                                
                                # Handle list format (segments with timestamps)
                                if isinstance(transcript_json, list) and transcript_json: 
                                    final_transcript = " ".join(seg.get("text","") for seg in transcript_json if "text" in seg)

                                # Handle complex nested structure
                                if not final_transcript and isinstance(transcript_json, dict): 
                                     if 'results' in transcript_json and isinstance(transcript_json['results'], list):
                                         try:
                                             final_transcript = transcript_json['results'][0]['alternatives'][0]['transcript']
                                         except (IndexError, KeyError, TypeError):
                                             logger.warning("Could not parse complex JSON transcript structure.")
                                             final_transcript = str(transcript_json) 
                                     else: 
                                        final_transcript = str(transcript_json)
                            except json.JSONDecodeError:
                                logger.error("Failed to decode JSON transcript. Using content as is.")
                                final_transcript = content
                        else:  # For .txt files or other formats
                            final_transcript = content
                        
                        logger.info("Transcription extracted.")
                    except Exception as e:
                        logger.error(f"Error reading transcript file {local_transcript_path}: {e}")
                    finally:
                        # Clean up temporary files
                        try:
                            os.remove(local_transcript_path)
                            if not os.listdir(temp_download_dir):
                                os.rmdir(temp_download_dir)
                        except OSError as e_clean:
                            logger.warning(f"Could not clean up temp transcript file/dir: {e_clean}")
                else:
                    logger.error(f"Failed to download transcript file: {transcript_file_name}")
            else:
                logger.error("Could not determine transcript file name from output storage.")
            break  # Exit the polling loop

        elif status in ["Failed", "Cancelled", "Terminated"]:
            logger.error(f"❌ Job failed with status: {status}. Reason: {job_status_response.get('error_message', 'N/A')}")
            break
        
        # If job is still in progress, continue polling

    if not final_transcript and status != "Completed":
        logger.error(f"Transcription job {job_id} did not complete successfully.")
    
    return final_transcript


# --- Question Answering with LLM ---
def get_answer_from_transcript_sarvam(transcript: str, question: str, conversation_history=None) -> str | None:
    """
    Get an answer to a question about the transcript using Sarvam AI's language model.
    
    Args:
        transcript: The full transcript text
        question: The question to answer
        conversation_history: Previous messages in the conversation (optional)
        
    Returns:
        The AI-generated answer, or None if the request failed
    """
    if not SARVAM_API_KEY:
        logger.error("Error: SARVAM_API_KEY not found in environment variables for LLM.")
        return None

    headers = {
        "api-subscription-key": f"{SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }

    # Include transcript in the system message to avoid multiple system messages
    system_content = """You are a helpful and friendly video tutor assistant. Your task is to have a natural conversation 
    about the video transcript provided. Answer questions based ONLY on the transcript content. 
    If the answer cannot be found in the transcript, say 'I don't see information about that in the video.'
    Keep your tone friendly, engaging and conversational.

    Here is the video transcript:
    """
    
    system_content += f"\"\"\"\n{transcript}\n\"\"\"\n"
    
    # Start with just one system message containing both instructions and transcript
    messages = [{"role": "system", "content": system_content}]
    
    # If no history or new conversation, just add the question
    if not conversation_history or len(conversation_history) == 0:
        messages.append({"role": "user", "content": question})
    else:
        # Add conversation history (up to the last 6 messages to keep context focused)
        history_to_include = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        messages.extend(history_to_include)
        
        # Make sure the last message is the current question if not already included
        if history_to_include and history_to_include[-1]["role"] != "user":
            messages.append({"role": "user", "content": question})

    # Format the request for the AI model
    payload = {
        "model": SARVAM_LLM_MODEL_NAME,
        "messages": messages,
    }

    try:
        logger.info(f"Sending question to Sarvam LLM (Model: {SARVAM_LLM_MODEL_NAME})...")
        response = requests.post(SARVAM_LLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        response_json = response.json()
        
        # Extract the answer text from the response
        answer_content = None
        if response_json.get("choices") and isinstance(response_json["choices"], list) and len(response_json["choices"]) > 0:
            message = response_json["choices"][0].get("message")
            if message and message.get("content"):
                answer_content = message["content"]
        elif response_json.get("generated_text"):  # Alternative response format
             answer_content = response_json.get("generated_text")
        
        if answer_content:
            # Ensure the answer is a single string
            if isinstance(answer_content, list):
                final_answer = "\n".join(str(item) for item in answer_content)
            elif isinstance(answer_content, str):
                final_answer = answer_content
            else:
                final_answer = str(answer_content)

            logger.info("LLM generated answer successfully.")
            return final_answer.strip()
        else:
            logger.error(f"LLM call succeeded but no answer content found in response")
            return "Error: Could not extract an answer from the LLM response."

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred with LLM API: {http_err}")
        logger.error(f"Response content: {response.content.decode()}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred with LLM API: {req_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred with LLM API: {e}")
        
    return None


# --- Real-time Speech-to-Text ---
def transcribe_audio_direct(audio_file_path: str, language: str = "en") -> str | None:
    """
    Transcribe a short audio clip directly (without batch processing).
    Good for transcribing user questions from microphone.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        language: Language code (e.g., "en" for English)
        
    Returns:
        The transcribed text, or None if transcription fails
    """
    if not SARVAM_API_KEY:
        logger.error("Error: SARVAM_API_KEY not found in environment variables for direct STT.")
        return None

    if not os.path.exists(audio_file_path):
        logger.error(f"Error: Audio file not found at {audio_file_path}")
        return None

    headers = {
        "api-subscription-key": f"{SARVAM_API_KEY}",
    }

    try:
        with open(audio_file_path, "rb") as f_audio:
            files = {
                "file": (os.path.basename(audio_file_path), f_audio, "audio/wav")
            }
            data = {
                "language_code": "unknown"  # Let the API auto-detect or specify a language
            }

            logger.info(f"Sending audio for direct transcription")
            response = requests.post(SARVAM_REALTIME_STT_API_URL, headers=headers, files=files, data=data)
            response.raise_for_status()
            
            response_json = response.json()
            
            # Extract the transcript text from the response
            transcript = response_json.get("transcript")

            if transcript:
                logger.info("Direct transcription successful.")
                return transcript
            else:
                logger.error(f"Direct transcription failed. No text found in response")
                return None

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred during direct STT: {http_err}")
        logger.error(f"Response content: {response.content.decode() if response else 'No response'}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred during direct STT: {req_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during direct STT: {e}")
        
    return None


# --- Text-to-Speech (Voice Generation) ---
def text_to_speech_sarvam(text_to_speak: str, output_audio_path: str = "generated_speech", 
                         output_filename: str = "response.wav", language_code: str = "en-IN",
                         voice_style: str = "neutral") -> str | None:
    """
    Convert text to spoken audio using Sarvam AI's TTS service.
    
    Args:
        text_to_speak: The text to convert to speech
        output_audio_path: Directory to save the audio file
        output_filename: Name of the output audio file
        language_code: Language code (e.g., "en-IN" for Indian English)
        voice_style: Style of voice (e.g., "neutral", "conversational")
        
    Returns:
        Path to the generated audio file, or None if TTS fails
    """
    if not SARVAM_API_KEY:
        logger.error("Error: SARVAM_API_KEY not found for TTS.")
        return None

    # Create output directory if it doesn't exist
    if not os.path.exists(output_audio_path):
        try:
            os.makedirs(output_audio_path)
            logger.info(f"Created directory for TTS output: {output_audio_path}")
        except OSError as e:
            logger.error(f"Error creating directory {output_audio_path}: {e}")
            return None
            
    final_output_file_path = os.path.join(output_audio_path, output_filename)

    headers = {
        "api-subscription-key": f"{SARVAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text_to_speak,
        "target_language_code": language_code,
        # Add voice_style if your API supports it
        # "voice_style": voice_style
    }

    # We'll save audio segments here temporarily
    temp_segment_files = []
    segment_temp_dir = os.path.join(output_audio_path, f"temp_segments_{int(time.time())}")

    try:
        logger.info(f"Sending text to TTS API (length: {len(text_to_speak)} chars)")
        response = requests.post(SARVAM_TTS_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        
        # The API returns a list of base64-encoded audio segments
        audios_list_base64 = response_json.get("audios")

        if not audios_list_base64 or not isinstance(audios_list_base64, list) or not audios_list_base64:
            logger.error(f"TTS API response did not contain valid 'audios' list")
            return None

        logger.info(f"Received {len(audios_list_base64)} audio segment(s) from TTS API.")
        
        # Create temp directory for audio segments
        if not os.path.exists(segment_temp_dir):
            os.makedirs(segment_temp_dir)

        # Process each audio segment
        for i, base64_audio_string in enumerate(audios_list_base64):
            if not isinstance(base64_audio_string, str):
                logger.warning(f"Segment {i} in 'audios' list is not a string. Skipping.")
                continue
            try:
                # Decode base64 to binary audio data
                audio_bytes = base64.b64decode(base64_audio_string)
                temp_segment_file_path = os.path.join(segment_temp_dir, f"segment_{i}.wav")
                with open(temp_segment_file_path, "wb") as f_segment:
                    f_segment.write(audio_bytes)
                temp_segment_files.append(temp_segment_file_path)
            except base64.binascii.Error as b64_err:
                logger.error(f"Failed to decode base64 for audio segment {i}: {b64_err}")
            except Exception as e_write:
                logger.error(f"Failed to write audio segment {i} to file: {e_write}")
        
        if not temp_segment_files:
            logger.error("No audio segments were successfully processed.")
            return None

        # If there's only one segment, just move it to the final location
        if len(temp_segment_files) == 1:
            logger.info("Only one audio segment, moving it to final destination.")
            shutil.move(temp_segment_files[0], final_output_file_path)
        else:
            # If multiple segments, concatenate them using FFmpeg
            logger.info(f"Combining {len(temp_segment_files)} audio segments with FFmpeg.")
            
            # Create a list file for FFmpeg
            file_list_path = os.path.join(segment_temp_dir, "ffmpeg_filelist.txt")
            with open(file_list_path, "w") as fl:
                for f_path in temp_segment_files:
                    clean_f_path = os.path.abspath(f_path)
                    fl.write(f"file '{clean_f_path}'\n")
            
            # Run FFmpeg to combine audio segments
            ffmpeg_command = [
                "ffmpeg", "-y",  # Overwrite output if it exists
                "-f", "concat",  # Use concat mode
                "-safe", "0",    # Allow absolute paths
                "-i", file_list_path, 
                "-c", "copy",    # Copy audio without re-encoding
                final_output_file_path
            ]
            
            process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False)

            if process.returncode == 0:
                logger.info(f"FFmpeg concatenation successful.")
            else:
                logger.error(f"FFmpeg concatenation failed. Error: {process.stderr}")
                return None

        return final_output_file_path

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred during TTS: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred during TTS: {req_err}")
    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to decode JSON response from TTS API: {json_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during TTS: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(segment_temp_dir):
            try:
                shutil.rmtree(segment_temp_dir)
            except Exception as e_clean:
                logger.error(f"Error cleaning up temporary files: {e_clean}")
        
    return None


# Test code that runs when this file is executed directly
if __name__ == '__main__':
    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tts_test_output_dir = os.path.join(project_root_dir, "test_tts_audio")

    print("\n--- Testing Text-to-Speech (TTS) ---")
    if SARVAM_API_KEY:
        sample_text = "Hello, this is a test of the Sarvam AI Text to Speech service, outputting base64 WAV."
        print(f"Text to speak: \"{sample_text}\"")
        
        generated_audio_file = text_to_speech_sarvam(
            sample_text, 
            output_audio_path=tts_test_output_dir,
            output_filename="sarvam_tts_test.wav"
        )
        
        if generated_audio_file and os.path.exists(generated_audio_file):
            print(f"TTS audio generated successfully: {generated_audio_file}")
        else:
            print("TTS audio generation failed.")
    else:
        print("Skipping TTS test: SARVAM_API_KEY not set in .env file.")