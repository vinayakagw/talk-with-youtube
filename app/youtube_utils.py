import yt_dlp
import os
import re

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    # Remove invalid characters like < > : " / \ | ? * and control characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Replace sequences of whitespace with a single underscore
    name = re.sub(r'\s+', '_', name)
    # Strip leading/trailing underscores that might result from multiple spaces at ends
    name = name.strip('_')
    # Limit length to avoid overly long filenames
    return name[:100]


def download_youtube_audio(video_url: str, output_path: str = "audio_files") -> str | None:
    """
    Downloads the audio from a YouTube video with a unique name.
    The final filename will be <output_path>/<sanitized_title>_<video_id>.mp3.

    Args:
        video_url: The URL of the YouTube video.
        output_path: The directory to save the audio file.

    Returns:
        The full path to the downloaded audio file, or None if download fails or file not verified.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    video_id = None
    video_title = "untitled_video" # Default title if extraction fails

    # Step 1: Extract video information (ID and title)
    # Using 'extract_flat': False to get title, 'skip_download': True to only fetch info
    info_opts = {'quiet': True, 'extract_flat': False, 'skip_download': True}
    try:
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_id = info_dict.get('id')
            video_title = info_dict.get('title', video_title) # Use fetched title or default
            if not video_id:
                print(f"Error: Could not extract video ID for URL {video_url}.")
                return None
    except Exception as e:
        print(f"Error extracting video info for {video_url}: {e}")
        return None

    # Step 2: Construct a unique base filename (without extension)
    sanitized_title = sanitize_filename(video_title)
    unique_base_name = f"{sanitized_title}_{video_id}"
    
    # This is the path template for yt-dlp (it will append .mp3 due to postprocessor)
    # e.g., /Users/../temp_audio/My_Video_Title_xxxxxxxxx
    outtmpl_path_without_ext = os.path.join(output_path, unique_base_name)
    
    # This is the final expected path after postprocessing
    # e.g., /Users/../temp_audio/My_Video_Title_xxxxxxxxx.mp3
    expected_final_mp3_path = outtmpl_path_without_ext + ".mp3"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outtmpl_path_without_ext, # Provide base path; postprocessor adds .mp3
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'verbose': True, # Keep for debugging for now; can be set to False or quiet: True later
        'overwrites': False, # Default is 'no_overwrites': True. yt-dlp won't overwrite existing files.
    }

    print(f"Attempting to download audio for '{video_title}' (ID: {video_id}).")
    print(f"Output template (without extension): {outtmpl_path_without_ext}")
    print(f"Expected final MP3 path: {expected_final_mp3_path}")

    # If an old zero-byte file exists, yt-dlp might not overwrite it or might error.
    # Let's remove it to ensure a clean download attempt.
    if os.path.exists(expected_final_mp3_path) and os.path.getsize(expected_final_mp3_path) == 0:
        print(f"Found zero-byte existing file at {expected_final_mp3_path}. Deleting it before download attempt.")
        try:
            os.remove(expected_final_mp3_path)
        except OSError as e:
            print(f"Warning: Could not delete zero-byte file {expected_final_mp3_path}: {e}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([video_url])
            
            if error_code == 0:
                print(f"yt-dlp download process for '{video_title}' (ID: {video_id}) completed with error code 0.")
                # Explicitly check if the file exists and is not empty
                if os.path.exists(expected_final_mp3_path) and os.path.getsize(expected_final_mp3_path) > 0:
                    print(f"Audio downloaded and verified successfully: {expected_final_mp3_path}")
                    return expected_final_mp3_path
                else:
                    print(f"Error: yt-dlp reported success (code 0) for '{video_title}', but expected MP3 file not found or is empty: {expected_final_mp3_path}.")
                    if os.path.exists(output_path):
                        print(f"Contents of directory '{output_path}': {os.listdir(output_path)}")
                    # Check if the non-extension path exists (original download before FFmpeg)
                    if os.path.exists(outtmpl_path_without_ext) and not outtmpl_path_without_ext.endswith(".mp3"):
                         print(f"A file might exist at the base outtmpl path (without .mp3): {outtmpl_path_without_ext}. FFmpeg conversion to MP3 might have failed.")
                    return None
            else:
                print(f"Error downloading audio for '{video_title}' (ID: {video_id}). yt-dlp returned error code: {error_code}")
                return None
    except Exception as e:
        print(f"An exception occurred during download with yt-dlp for '{video_title}' (ID: {video_id}): {e}")
        return None