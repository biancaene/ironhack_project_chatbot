# download_main.py
from config import CHANNEL_ID, OUTPUT_DIR, VIDEO_LIMIT, DELAY_MIN, DELAY_MAX

import os
import re
import time
import random
from yt_dlp import YoutubeDL

from channel import get_latest_videos_from_channel, get_videos_from_channel 
from download_video import download_video
from download_transcript import download_transcript
from generate_transcript import transcribe_with_whisper


def find_video_file(output_dir, video_title):
    """Find mp4 file downloaded for a video."""
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', video_title)
    for f in os.listdir(output_dir):
        if f.endswith(".mp4") and safe_title[:30] in f:
            return os.path.join(output_dir, f)
    return None


def main():
    languages = ['en', 'ro']

    print("Fetching latest videos...")
    
    videos = get_videos_from_channel(CHANNEL_ID, VIDEO_LIMIT)
    print("videos = ", videos)

    for i, url in enumerate(videos):
        print(f"\nProcessing {url}")
        try:
            info = download_video(url, OUTPUT_DIR)

            video_id = info["id"]
            title = info["title"]

            # Try YouTube transcript download for all languages at once
            transcript_files = download_transcript(video_id, title, OUTPUT_DIR, languages)

            # Check which languages are missing transcripts
            missing_languages = []
            for lang in languages:
                lang_has_transcript = any(lang in str(f) for f in (transcript_files or []))
                if not lang_has_transcript:
                    missing_languages.append(lang)

            if missing_languages:
                print(f"Missing transcripts for languages {missing_languages} in {video_id}, trying Whisper...")

                video_path = find_video_file(OUTPUT_DIR, title)

                if video_path:
                    for lang in missing_languages:
                        print(f"Generating Whisper transcript for language: {lang}")
                        whisper_file = transcribe_with_whisper(video_id, title, OUTPUT_DIR, video_path, lang)
                        if whisper_file:
                            print(f"Whisper transcript created for [{lang}]: {whisper_file}")
                        else:
                            print(f"Whisper failed for {video_id} [{lang}], skipping.")
                else:
                    print(f"Video file not found for {title}, cannot transcribe.")

        except Exception as e:
            print(f"Error: {e}")

        if i < len(videos) - 1:
            delay = random.randint(DELAY_MIN, DELAY_MAX)
            print(f"Sleeping {delay} seconds")
            time.sleep(delay)


if __name__ == "__main__":
    main()