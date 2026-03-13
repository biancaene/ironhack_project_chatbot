# query_agent.py
import os
import sys
os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from video_player import get_stream_url

import time
import vlc

from dotenv import load_dotenv, find_dotenv

from rag_core import run_rag  # RAG (answer + segments)

# =========================
# Load environment variables
# =========================
_ = load_dotenv(find_dotenv())


print("Ask me anything about the YouTube channel videos (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # =========================
    # RAG: answer + segmente video
    # =========================
    result = run_rag(user_input)

    answer = result["answer"]
    segments = result["segments"]

    print("\nAssistant:\n", answer)

    if not segments:
        print("\nNo video segments found.")
        continue

    print("\nVideo sources:")
    for i, s in enumerate(segments, start=1):
        #print(f"{i}. {s['start_time']} -> {s['youtube_link']}")
        print(f"{i}. {s['start_time']} -> {s['watch_link']}")
        

    # =========================
    # VLC playback for each segment
    # =========================
    for s in segments:
        url = s["watch_link"]

        # direct YouTube link to play
        #media_url = url # not working with VLC, YouTube blocks direct playback, so we need to get the stream URL with yt_dlp or similar tool

        # stream URL with yt_dlp:
        try:
            media_url = get_stream_url(url)
        except Exception as e:
            print(f"Could not get stream URL for {url}: {e}")
            print("Skipping this segment...")
            continue

        print(f"\nPlaying segment: {s['start_time']} -> {media_url}")

        player = vlc.MediaPlayer(media_url)
        player.play()

        # small delay to start the player
        time.sleep(1)

        # set the time for starting the stream (în milisecunde)
        player.set_time(s["seconds"] * 1000)

        input("Press Enter to stop this segment and go to the next...")
        player.stop()
