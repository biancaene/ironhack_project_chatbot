import os
from yt_dlp import YoutubeDL


# =========================
# download video
# =========================
def download_video(video_url, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- 1. Download video + audio ----------------
    ydl_opts_video = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'download_archive': 'downloaded.txt',
        #'remote_components': 'ejs:github',
        #'extractor_args': {
        #    'youtube': {
        #        'js_runtimes': ['node']
        #    }
        #},
        #'cookiesfrombrowser': ('edge',),
        "cookiesfile": "cookies.txt"
    }

    # get info 
    ydl_opts_info = {
        'quiet': True,
        'skip_download': True,
        'cookiesfile': 'cookies.txt'
    }

    with YoutubeDL(ydl_opts_video) as ydl:
        ydl.download([video_url])
        
    with YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(video_url, download=False)
    
    return info