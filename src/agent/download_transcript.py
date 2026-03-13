# download_transcript.py
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


#print(dir(YouTubeTranscriptApi))


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


# =========================
# download transcript
# =========================
def download_transcript(video_id, video_title, output_dir, languages=['en', 'ro']):
    
    os.makedirs(output_dir, exist_ok=True)
    
    transcript_files  = []

    for lang in languages:
        try:
            safe_title = sanitize_filename(video_title)
            
            transcript_file = os.path.join(output_dir, f"{safe_title}_transcript_{lang}_{video_id}.txt")

            # skip dacă există deja
            if os.path.exists(transcript_file):
                print(f"Transcript {lang} already exists: {transcript_file}")
                transcript_files.append(transcript_file)
                continue

            transcript = YouTubeTranscriptApi().fetch(video_id, languages=[lang])
            
            with open(transcript_file, "w", encoding="utf-8") as f:
                for entry in transcript:
                    start = entry.start  # secunde
                    text = entry.text

                    # convert seconds -> hh:mm:ss
                    hours = int(start // 3600)
                    minutes = int((start % 3600) // 60)
                    seconds = int(start % 60)
                    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                    f.write(f"[{timestamp}] {text}\n")
                f.write("\n")


            transcript_files.append(transcript_file)
            print(f"Transcript {lang} saved: {transcript_file}")

        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"Transcript in {lang} was not found.")

    return transcript_files