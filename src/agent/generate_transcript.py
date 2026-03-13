# generate_transcript.py
import os
import re
import subprocess
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
client = OpenAI()


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def extract_audio_for_whisper(video_path, output_dir):
    """
    Extracts compressed audio from video file, keeping it under 25MB.
    Returns path to audio file, or None if failed.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}_audio.mp3")

    if os.path.exists(audio_path):
        print(f"Audio already extracted: {audio_path}")
        return audio_path

    print(f"Extracting audio from: {video_path}")

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",              # no video
            "-ar", "16000",     # 16kHz sample rate (sufficient for speech)
            "-ac", "1",         # mono
            "-b:a", "32k",      # low bitrate to keep file small
            audio_path
        ], check=True, capture_output=True)

        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"Audio extracted: {audio_path} ({size_mb:.1f} MB)")

        if size_mb > 25:
            print(f"WARNING: Audio still {size_mb:.1f} MB, exceeds Whisper 25MB limit!")
            return None

        return audio_path

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e.stderr.decode()}")
        return None


def transcribe_with_whisper(video_id, video_title, output_dir, video_path, lang=None, retries=3):
    os.makedirs(output_dir, exist_ok=True)

    safe_title = sanitize_filename(video_title)
    lang_suffix = f"_{lang}" if lang else "_whisper"
    transcript_file = os.path.join(output_dir, f"{safe_title}_transcript{lang_suffix}_{video_id}.txt")

    if os.path.exists(transcript_file):
        print(f"Whisper transcript already exists: {transcript_file}")
        return transcript_file

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None

    # Extract compressed audio to stay under 25MB Whisper limit
    audio_path = extract_audio_for_whisper(video_path, output_dir)
    if not audio_path:
        return None

    print(f"Transcribing with Whisper: {audio_path}")

    transcription_params = {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"]
    }
    if lang:
        transcription_params["language"] = lang

    for attempt in range(retries):
        try:
            with open(audio_path, "rb") as f:
                transcription_params["file"] = f
                response = client.audio.transcriptions.create(**transcription_params)
            break

        except Exception as e:
            print(f"Whisper attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                print(f"Whisper transcription failed after {retries} attempts.")
                # cleanup audio file
                os.remove(audio_path)
                return None

    # cleanup temporary audio file
    os.remove(audio_path)

    with open(transcript_file, "w", encoding="utf-8") as out:
        for segment in response.segments:
            start = segment.start
            hours   = int(start // 3600)
            minutes = int((start % 3600) // 60)
            seconds = int(start % 60)
            timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            text = segment.text.strip()
            out.write(f"[{timestamp}] {text}\n")
        out.write("\n")

    print(f"Whisper transcript saved: {transcript_file}")
    return transcript_file