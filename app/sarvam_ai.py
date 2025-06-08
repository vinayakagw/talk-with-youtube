import asyncio
import aiofiles
import requests
import json
import os
import time
import mimetypes
import logging
from urllib.parse import urlparse
from azure.storage.filedatalake.aio import DataLakeDirectoryClient, FileSystemClient
from azure.storage.filedatalake import ContentSettings
from dotenv import load_dotenv
from pprint import pprint
import base64
import subprocess # For calling FFmpeg
import shutil # For moving files

# Load environment variables from .env file
# Ensure .env is in the project root, not in the 'app' directory for this load_dotenv call
# Or provide a specific path: load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# API Endpoints for Batch STT
SARVAM_JOB_INIT_URL = "https://api.sarvam.ai/speech-to-text/job/init"
SARVAM_JOB_START_URL = "https://api.sarvam.ai/speech-to-text/job" # POST to start
SARVAM_JOB_STATUS_URL_TEMPLATE = "https://api.sarvam.ai/speech-to-text/job/{job_id}/status" # GET for status

# LLM, Realtime STT, and TTS URLs/Model Names
SARVAM_LLM_API_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_LLM_MODEL_NAME = "sarvam-m"
SARVAM_REALTIME_STT_API_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_API_URL = "https://api.sarvam.ai/text-to-speech"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SarvamStorageClient:
    def __init__(self, storage_url_with_sas: str):
        self.account_url, self.file_system_name, self.directory_name, self.sas_token = (
            self._extract_url_components(storage_url_with_sas)
        )
        self.lock = asyncio.Lock()
        logger.info(f"Initialized SarvamStorageClient with directory: {self.directory_name}")

    def update_storage_url(self, storage_url_with_sas: str):
        self.account_url, self.file_system_name, self.directory_name, self.sas_token = (
            self._extract_url_components(storage_url_with_sas)
        )
        logger.info(f"Updated SarvamStorageClient URL to directory: {self.directory_name}")

    def _extract_url_components(self, url: str):
        parsed_url = urlparse(url)
        account_url = f"{parsed_url.scheme}://{parsed_url.netloc}".replace(
            ".blob.", ".dfs." # Important for DataLake
        )
        path_components = parsed_url.path.strip("/").split("/")
        file_system_name = path_components[0]
        directory_name = "/".join(path_components[1:])
        sas_token = parsed_url.query
        return account_url, file_system_name, directory_name, sas_token

    async def upload_file(self, local_file_path: str, overwrite: bool = True) -> bool:
        logger.info(f"Starting upload of file: {local_file_path}")
        async with DataLakeDirectoryClient(
            account_url=f"{self.account_url}?{self.sas_token}",
            file_system_name=self.file_system_name,
            directory_name=self.directory_name,
            credential=None, # SAS token is in the URL
        ) as directory_client:
            file_name = os.path.basename(local_file_path)
            try:
                async with aiofiles.open(local_file_path, mode="rb") as file_data:
                    mime_type, _ = mimetypes.guess_type(local_file_path)
                    if mime_type is None:
                        if local_file_path.lower().endswith(".mp3"):
                            mime_type = "audio/mpeg"
                        else:
                            mime_type = "application/octet-stream" # Default
                    
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
        logger.info(f"\n📂 Listing files in directory: {self.directory_name}...")
        file_names = []
        async with FileSystemClient(
            account_url=f"{self.account_url}?{self.sas_token}",
            file_system_name=self.file_system_name,
            credential=None,
        ) as file_system_client:
            async for path in file_system_client.get_paths(self.directory_name):
                # path.name includes the full path from the filesystem root, e.g., "dirname/filename.txt"
                # We only want the filename if directory_name is not empty.
                if self.directory_name and path.name.startswith(self.directory_name + "/"):
                    file_name = path.name[len(self.directory_name)+1:]
                else: # If directory_name is empty or path is at root of filesystem
                    file_name = path.name
                
                if not path.is_directory: # We only want files
                    async with self.lock: # Just in case, though list_files is usually called once
                        file_names.append(file_name)
        logger.info(f"Found {len(file_names)} files: {file_names}")
        return file_names

    async def download_file(self, file_name: str, destination_local_path: str) -> bool:
        logger.info(f"\n⬇️ Starting download of {file_name} to {destination_local_path}")
        # Ensure destination directory exists
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

# --- Batch Job Management Functions (Synchronous HTTP calls) ---
def _get_headers():
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not found in environment variables.")
    return {
        "API-Subscription-Key": SARVAM_API_KEY,
        "Content-Type": "application/json", # Usually for POST/PUT with JSON body
    }

def initialize_stt_job():
    logger.info("🚀 Initializing STT job...")
    headers = _get_headers()
    # Remove Content-Type for this specific POST if it doesn't have a body, or set as needed by API
    if "Content-Type" in headers and SARVAM_JOB_INIT_URL == "https://api.sarvam.ai/speech-to-text/job/init": # Example check
         # The example notebook call to init doesn't show a Content-Type, requests might set one.
         # Let's use a minimal header for init.
         init_headers = {"API-Subscription-Key": SARVAM_API_KEY}

    response = requests.post(SARVAM_JOB_INIT_URL, headers=init_headers)
    logger.info(f"Initialize Job Response Status: {response.status_code}")
    if response.status_code == 202: # Accepted
        job_info = response.json()
        logger.info("Job initialized successfully:")
        pprint(job_info)
        return job_info
    else:
        logger.error(f"Job initialization failed: {response.status_code} - {response.text}")
        return None

def start_stt_job(job_id: str, language_code: str = "unknown"): # Default to en-US, adjust as needed
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
    if response.status_code == 200: # OK
        job_start_info = response.json()
        logger.info("Job started successfully:")
        pprint(job_start_info)
        return job_start_info
    else:
        logger.error(f"Failed to start job: {response.status_code} - {response.text}")
        return None

def check_stt_job_status(job_id: str):
    logger.info(f"🔍 Checking status for job: {job_id}")
    url = SARVAM_JOB_STATUS_URL_TEMPLATE.format(job_id=job_id)
    headers = _get_headers()
    # GET request typically doesn't need Content-Type in headers
    status_headers = {"API-Subscription-Key": SARVAM_API_KEY}
    response = requests.get(url, headers=status_headers)
    logger.info(f"Job Status Response Status: {response.status_code}")
    if response.status_code == 200: # OK
        job_status = response.json()
        logger.info("Job status:")
        pprint(job_status)
        return job_status
    else:
        logger.error(f"Failed to get job status: {response.status_code} - {response.text}")
        return None

async def transcribe_audio_sarvam_batch(audio_file_path: str) -> str | None:
    """
    Transcribes an audio file using Sarvam AI's Batch STT API.
    Orchestrates upload, job creation, monitoring, and result download.
    """
    if not SARVAM_API_KEY:
        logger.error("Error: SARVAM_API_KEY not found in environment variables.")
        return None
    if not os.path.exists(audio_file_path):
        logger.error(f"Error: Audio file not found at {audio_file_path}")
        return None


    # 1. Initialize Job
    job_info = initialize_stt_job()
    if not job_info or "job_id" not in job_info:
        return None
    
    job_id = job_info["job_id"]
    input_storage_path_sas = job_info.get("input_storage_path")
    output_storage_path_sas = job_info.get("output_storage_path")

    if not input_storage_path_sas or not output_storage_path_sas:
        logger.error("Missing input_storage_path or output_storage_path in job initialization response.")
        return None

    # 2. Upload audio file to input storage
    input_client = SarvamStorageClient(input_storage_path_sas)
    upload_successful = await input_client.upload_file(audio_file_path)
    if not upload_successful:
        logger.error(f"Failed to upload {audio_file_path} to Sarvam storage.")
        return None

    # 3. Start the STT job
    start_response = start_stt_job(job_id, language_code="unknown")
    if not start_response: # or start_response.get("status") != "Started" (check actual response)
        logger.error(f"Failed to start STT job {job_id}.")
        return None

    # 4. Monitor job status
    logger.info(f"⏳ Monitoring job status for {job_id}...")
    max_retries = 30 # e.g., 30 retries
    retry_delay_seconds = 20 # e.g., 20 seconds delay
    
    final_transcript = None

    for attempt in range(max_retries):
        await asyncio.sleep(retry_delay_seconds) # Use asyncio.sleep in async function
        logger.info(f"Status check attempt {attempt + 1}/{max_retries} for job {job_id}")
        job_status_response = check_stt_job_status(job_id)

        if not job_status_response:
            logger.warning("Failed to get job status on this attempt.")
            continue # Try again after delay

        status = job_status_response.get("job_state") # Or "status", "state" - check actual response key
        logger.info(f"Current job state: {status}")

        if status == "Completed":
            logger.info("✅ Job completed successfully!")
            # 5. Download results
            output_client = SarvamStorageClient(output_storage_path_sas)
            
            # List files in the output directory to find the transcript file
            output_files = await output_client.list_files()
            if not output_files:
                logger.error("No files found in the output storage.")
                return None

            # Assuming the transcript is in a .json or .txt file,
            # often named similarly to the input or a standard name.
            # This part might need adjustment based on actual Sarvam output.
            transcript_file_name = None
            input_base_name = os.path.splitext(os.path.basename(audio_file_path))[0]

            for fname in output_files:
                # Prioritize JSON, then TXT. Look for files related to input name or common names.
                if fname.lower().endswith(".json"): # Common for detailed transcripts
                    transcript_file_name = fname
                    break
            if not transcript_file_name: # Fallback to .txt
                 for fname in output_files:
                    if fname.lower().endswith(".txt"):
                        transcript_file_name = fname
                        break
            
            if not transcript_file_name and output_files: # If no specific match, take the first non-log file
                potential_files = [f for f in output_files if not f.lower().endswith(".log")]
                if potential_files:
                    transcript_file_name = potential_files[0]
                else: # Or just the first file if only logs are present (unlikely for transcript)
                    transcript_file_name = output_files[0]


            if transcript_file_name:
                temp_download_dir = os.path.join(project_root, "temp_transcripts")
                os.makedirs(temp_download_dir, exist_ok=True)
                local_transcript_path = os.path.join(temp_download_dir, transcript_file_name)
                
                download_ok = await output_client.download_file(transcript_file_name, local_transcript_path)
                if download_ok and os.path.exists(local_transcript_path):
                    logger.info(f"Transcript file downloaded to: {local_transcript_path}")
                    try:
                        async with aiofiles.open(local_transcript_path, "r", encoding="utf-8") as f:
                            content = await f.read()
                        # Parse content: if JSON, look for 'transcript' or 'text' field. If TXT, use as is.
                        if transcript_file_name.lower().endswith(".json"):
                            try:
                                transcript_json = json.loads(content)
                                # Common keys for transcript text:
                                final_transcript = transcript_json.get("transcript") or \
                                                   transcript_json.get("text") or \
                                                   transcript_json.get("full_transcript") 
                                if isinstance(transcript_json, list) and transcript_json: 
                                    final_transcript = " ".join(seg.get("text","") for seg in transcript_json if "text" in seg)

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
                                final_transcript = content # Use raw content if not valid JSON
                        else: # For .txt files or other formats
                            final_transcript = content
                        
                        logger.info("Transcription extracted.")
                    except Exception as e:
                        logger.error(f"Error reading/parsing downloaded transcript file {local_transcript_path}: {e}")
                    finally:
                        # Clean up downloaded file
                        try:
                            os.remove(local_transcript_path)
                            # Try to remove dir if empty
                            if not os.listdir(temp_download_dir):
                                os.rmdir(temp_download_dir)
                        except OSError as e_clean:
                            logger.warning(f"Could not clean up temp transcript file/dir: {e_clean}")
                else:
                    logger.error(f"Failed to download or find transcript file: {transcript_file_name}")
            else:
                logger.error("Could not determine transcript file name from output storage.")
            break # Exit monitoring loop

        elif status in ["Failed", "Cancelled", "Terminated"]: # Add other terminal failure states
            logger.error(f"❌ Job failed with status: {status}. Reason: {job_status_response.get('error_message', 'N/A')}")
            break # Exit monitoring loop
        
        # Else, job is still "Pending", "Running", "Queued", etc. Continue polling.

    if not final_transcript and status != "Completed":
        logger.error(f"Transcription job {job_id} did not complete successfully after {max_retries} retries or failed.")
    
    return final_transcript


def get_answer_from_transcript_sarvam(transcript: str, question: str) -> str | None:
    """
    Sends a transcript and a question to Sarvam AI's LLM and gets an answer.

    Args:
        transcript: The transcript text.
        question: The question asked by the user.

    Returns:
        The answer from the LLM, or None if it fails.
    """
    global SARVAM_API_KEY # Ensure API key is accessible
    if not SARVAM_API_KEY:
        logger.error("Error: SARVAM_API_KEY not found in environment variables for LLM.")
        return None

    headers = {
        "api-subscription-key": f"{SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }

    # Construct the prompt. Adjust this based on how you want to instruct the LLM.
    # It's crucial to instruct the LLM to answer *only* based on the provided transcript.
    system_prompt = "You are a helpful assistant. Your task is to answer questions based ONLY on the provided transcript. If the answer cannot be found in the transcript, say 'The answer is not found in the transcript.'"
    user_content = f"Transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\nQuestion: {question}"

    # This payload structure is common for chat-based LLMs.
    # Verify the exact payload structure required by Sarvam AI's LLM API.
    payload = {
        "model": SARVAM_LLM_MODEL_NAME, # Specify the model you are using
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        # Add other parameters like temperature, max_tokens if needed
        # "temperature": 0.7,
        # "max_tokens": 500,
    }

    try:
        logger.info(f"Sending question to Sarvam LLM (Model: {SARVAM_LLM_MODEL_NAME}, URL: {SARVAM_LLM_API_URL})...")
        # print(f"Sending question to Sarvam LLM (Model: {SARVAM_LLM_MODEL_NAME}, URL: {SARVAM_LLM_API_URL})...")
        # print(f"Payload: {payload}") # For debugging the payload

        response = requests.post(SARVAM_LLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        response_json = response.json()
        
        answer_content = None
        if response_json.get("choices") and isinstance(response_json["choices"], list) and len(response_json["choices"]) > 0:
            message = response_json["choices"][0].get("message")
            if message and message.get("content"):
                answer_content = message["content"]
        elif response_json.get("generated_text"): # Fallback
             answer_content = response_json.get("generated_text")
        
        if answer_content:
            # Ensure the answer is a single string
            if isinstance(answer_content, list):
                final_answer = "\n".join(str(item) for item in answer_content) # Join list elements
            elif isinstance(answer_content, str):
                final_answer = answer_content
            else: # If it's some other type, try to convert to string
                final_answer = str(answer_content)

            logger.info("LLM generated answer successfully.")
            return final_answer.strip()
        else:
            logger.error(f"LLM call succeeded but no answer content found in response: {response_json}")
            return "Error: Could not extract an answer from the LLM response."

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred with LLM API: {http_err}")
        logger.error(f"Response content: {response.content.decode()}")
        # print(f"HTTP error occurred with LLM API: {http_err}")
        # print(f"Response content: {response.content.decode()}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred with LLM API: {req_err}")
        # print(f"Request exception occurred with LLM API: {req_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred with LLM API: {e}")
        # print(f"An unexpected error occurred with LLM API: {e}")
        
    return None

def transcribe_audio_direct(audio_file_path: str, language: str = "en") -> str | None:
    """
    Transcribes an audio file using Sarvam AI's direct STT API.

    Args:
        audio_file_path: The path to the audio file (e.g., WAV) to be transcribed.
        language: The language of the audio (e.g., "en", "hi"). Defaults to "en".

    Returns:
        The transcribed text, or None if transcription fails.
    """
    global SARVAM_API_KEY # Ensure API key is accessible
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
                "file": (os.path.basename(audio_file_path), f_audio, "audio/wav") # Assuming WAV, adjust if other format
            }
            data = {
                "language_code": "unknown"
            }

            logger.info(f"Sending {audio_file_path} for direct transcription to {SARVAM_REALTIME_STT_API_URL} ")
            response = requests.post(SARVAM_REALTIME_STT_API_URL, headers=headers, files=files, data=data)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            
            response_json = response.json()
            
            # Based on Sarvam STT API tutorial, transcript is in response_json['text']
            transcript = response_json.get("transcript")

            if transcript:
                logger.info("Direct transcription successful.")
                return transcript
            else:
                logger.error(f"Direct transcription failed. No text found in response: {response_json}")
                return None

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred during direct STT: {http_err}")
        logger.error(f"Response content: {response.content.decode() if response else 'No response'}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred during direct STT: {req_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during direct STT: {e}")
        
    return None

def text_to_speech_sarvam(text_to_speak: str, output_audio_path: str = "generated_speech", output_filename: str = "response.wav", language_code: str = "en-IN") -> str | None:
    """
    Converts text to speech using Sarvam AI's TTS API.
    If the API returns multiple audio segments (base64 encoded WAV),
    they are saved as temporary files and concatenated using FFmpeg.

    Args:
        text_to_speak: The text to be converted to speech.
        output_audio_path: The directory to save the generated audio file.
        output_filename: The name for the output WAV audio file.
        language_code: The language code for TTS (e.g., "en-IN", "hi-IN").

    Returns:
        The full path to the saved audio file, or None if TTS fails.
    """
    global SARVAM_API_KEY
    if not SARVAM_API_KEY:
        logger.error("Error: SARVAM_API_KEY not found for TTS.")
        return None

    
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
    }

    temp_segment_files = []
    # Create a temporary directory for segments within the output_audio_path
    # This helps in cleaning up and avoids cluttering the main audio path during processing.
    segment_temp_dir = os.path.join(output_audio_path, f"temp_segments_{int(time.time())}")

    try:
        logger.info(f"Sending text for TTS (length: {len(text_to_speak)} chars, lang: {language_code}) to Sarvam AI: \"{text_to_speak[:70]}...\"")
        response = requests.post(SARVAM_TTS_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        logger.debug(f"TTS API Full Response JSON: {json.dumps(response_json)}")
        
        audios_list_base64 = response_json.get("audios")

        if not audios_list_base64 or not isinstance(audios_list_base64, list) or not audios_list_base64:
            logger.error(f"TTS API response did not contain a valid 'audios' list or the list was empty. Response: {response_json}")
            return None

        logger.info(f"Received {len(audios_list_base64)} audio segment(s) from TTS API.")
        
        if not os.path.exists(segment_temp_dir):
            os.makedirs(segment_temp_dir)

        for i, base64_audio_string in enumerate(audios_list_base64):
            if not isinstance(base64_audio_string, str):
                logger.warning(f"Segment {i} in 'audios' list is not a string. Skipping. Found: {type(base64_audio_string)}")
                continue
            try:
                audio_bytes = base64.b64decode(base64_audio_string)
                temp_segment_file_path = os.path.join(segment_temp_dir, f"segment_{i}.wav")
                with open(temp_segment_file_path, "wb") as f_segment:
                    f_segment.write(audio_bytes)
                temp_segment_files.append(temp_segment_file_path)
                logger.debug(f"Saved temporary audio segment {i+1} to {temp_segment_file_path}")
            except base64.binascii.Error as b64_err:
                logger.error(f"Failed to decode base64 for audio segment {i}: {b64_err}")
            except Exception as e_write:
                logger.error(f"Failed to write audio segment {i} to file: {e_write}")
        
        if not temp_segment_files:
            logger.error("No audio segments were successfully processed and saved.")
            return None

        if len(temp_segment_files) == 1:
            logger.info("Only one audio segment received, moving it to final destination.")
            shutil.move(temp_segment_files[0], final_output_file_path)
        else:
            logger.info(f"Multiple audio segments ({len(temp_segment_files)}), concatenating with FFmpeg.")
            # Create a file list for FFmpeg's concat demuxer
            file_list_path = os.path.join(segment_temp_dir, "ffmpeg_filelist.txt")
            with open(file_list_path, "w") as fl:
                for f_path in temp_segment_files:
                    # FFmpeg concat demuxer needs relative paths if -safe 0 is not used,
                    # or absolute paths. Using absolute paths is safer here.
                    # Ensure paths are correctly escaped if they contain special characters,
                    # though os.path.join should handle basic path construction.
                    # For 'file' directive, paths should be single-quoted if they contain special chars.
                    # However, Python's subprocess list-based command usually handles spaces.
                    # Let's ensure paths are clean.
                    clean_f_path = os.path.abspath(f_path)
                    fl.write(f"file '{clean_f_path}'\n")
            
            # FFmpeg command to concatenate
            # -y: overwrite output file if it exists
            # -f concat: use the concat demuxer
            # -safe 0: allow unsafe file paths (though we use abspath, good for robustness)
            # -i file_list_path: input file list
            # -c copy: copy codecs (assuming all segments are compatible WAV)
            ffmpeg_command = [
                "ffmpeg", "-y", 
                "-f", "concat", 
                "-safe", "0", 
                "-i", file_list_path, 
                "-c", "copy", 
                final_output_file_path
            ]
            logger.debug(f"Executing FFmpeg command: {' '.join(ffmpeg_command)}")
            process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False)

            if process.returncode == 0:
                logger.info(f"FFmpeg concatenation successful. Output: {final_output_file_path}")
            else:
                logger.error(f"FFmpeg concatenation failed. Return code: {process.returncode}")
                logger.error(f"FFmpeg stdout: {process.stdout}")
                logger.error(f"FFmpeg stderr: {process.stderr}")
                return None # Concatenation failed

        return final_output_file_path

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred during TTS: {http_err}")
        if response is not None: 
            logger.error(f"Response status code: {response.status_code}, Response content: {response.content.decode() if response.content else 'No response content'}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred during TTS: {req_err}")
    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to decode JSON response from TTS API: {json_err}")
        if response is not None:
             logger.error(f"Non-JSON Response content: {response.text[:500]}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during TTS: {e}")
    finally:
        # Clean up temporary segment files and directory
        if os.path.exists(segment_temp_dir):
            try:
                shutil.rmtree(segment_temp_dir)
                logger.debug(f"Cleaned up temporary segment directory: {segment_temp_dir}")
            except Exception as e_clean:
                logger.error(f"Error cleaning up temporary segment directory {segment_temp_dir}: {e_clean}")
        
    return None


if __name__ == '__main__':
    # ... (existing __main__ tests for STT and LLM) ...
    # load_dotenv() # Already called at the top of the module
    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    tts_test_output_dir = os.path.join(project_root_dir, "test_tts_audio")


    print("\n--- Testing Text-to-Speech (TTS) ---")
    # SARVAM_TTS_API_URL is now hardcoded, so the check for placeholder can be removed
    if SARVAM_API_KEY: # Check only for API Key now for this test
        sample_text = "Hello, this is a test of the Sarvam AI Text to Speech service, outputting base64 WAV."
        print(f"Text to speak: \"{sample_text}\"")
        
        generated_audio_file = text_to_speech_sarvam(
            sample_text, 
            output_audio_path=tts_test_output_dir,
            output_filename="sarvam_tts_test.wav" # Changed to .wav
        )
        
        if generated_audio_file and os.path.exists(generated_audio_file):
            print(f"TTS audio generated successfully: {generated_audio_file}")
            # You can play this WAV file to verify.
        else:
            print("TTS audio generation failed.")
    else:
        print("Skipping TTS test: SARVAM_API_KEY not set in .env file.")