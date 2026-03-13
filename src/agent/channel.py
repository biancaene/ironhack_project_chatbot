# channel.py
import feedparser
from yt_dlp import YoutubeDL


# =========================
# Get latest videos from channel
# =========================
def get_latest_videos_from_channel(channel_id, limit):

    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

    feed = feedparser.parse(rss_url)

    urls = []

    ydl_opts = {
        "quiet": True,
        "skip_download": True
    }

    with YoutubeDL(ydl_opts) as ydl:

        for entry in feed.entries:

            url = entry.link

            info = ydl.extract_info(url, download=False)

            duration = info.get("duration", 0)

            # filter shorts (< 120 sec)
            if duration and duration >= 120 and duration <= 3600:
                urls.append(url)

            if len(urls) >= limit:
                break

    return urls

def get_videos_from_channel(channel_id, limit):
    
    urls = []
    
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,  # nu descarca, doar listează
        "playlistend": limit,
    }

    channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"

    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(channel_url, download=False)
        
        if result and "entries" in result:
            for entry in result["entries"]:
                if entry:
                    urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
                if len(urls) >= limit:
                    break

    return urls