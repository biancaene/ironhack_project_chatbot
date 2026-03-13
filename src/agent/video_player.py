# video_player.py
import yt_dlp


def get_stream_url(youtube_url):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "format": "best",
        #"js_runtimes": {
        #    "node": {
        #        "path": "C:\\Program Files\\nodejs\\node.exe"
        #    } 
        #}
        #"cookiesfrombrowser": ("edge",)
        #"cookiesfile": "cookies.txt"
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info["url"]
    except Exception as e:
        raise Exception(f"yt-dlp failed: {e}")


def play_video_segment(url: str, start: int, end: int) -> dict:
    """
    Returns a UI payload:
    - stream_url: direct link for VLC
    - start: start second
    - end: end second
    - embed_link: YouTube link with timestamp
    """
    try:
        stream_url = get_stream_url(url)
        print("stream_url = ", stream_url)
    except Exception as e:
        stream_url = None

    embed_link = f"{url}&t={start}s"

    return {
        "stream_url": stream_url,
        "start": start,
        "end": end,
        "embed_link": embed_link
    }
