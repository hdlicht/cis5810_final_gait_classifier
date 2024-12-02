from pytube import YouTube
import os

import yt_dlp

def download_youtube_video(url, output_dir):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }
    try:
    
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
urls = ["https://www.youtube.com/watch?v=yULxvDc9e8c",
"https://www.youtube.com/watch?v=zMeKiOtDG9I",
"https://www.youtube.com/watch?v=XYw9gQkQv_Y",
"https://www.youtube.com/watch?v=gEQ9je2hiGE",
"https://www.youtube.com/watch?v=esn4r6L-rhQ",
"https://www.youtube.com/watch?v=gXws-A4op-E",
"https://www.youtube.com/watch?v=OoCDFmCm1DE",
"https://www.youtube.com/watch?v=rySW_DzPVCw",
"https://www.youtube.com/watch?v=yFBy0X0D-w8",
"https://www.youtube.com/watch?v=JSyLnt3rLxs",
"https://www.youtube.com/watch?v=mjnL8jRppxg",
"https://www.youtube.com/watch?v=H7WThdv-fCQ",
"https://www.youtube.com/watch?v=jzkn287X-84",
"https://www.youtube.com/watch?v=dxRMtNtjwCc",
"https://www.youtube.com/watch?v=gpNLTB58kK0",
"https://www.youtube.com/watch?v=as0e_s4LMKE",
"https://www.youtube.com/watch?v=RnGsGqxJS-8",
"https://www.youtube.com/watch?v=YjRoLtP1di0",
"https://www.youtube.com/watch?v=WWS-iOlLsoo"]

output_dir = "videos/videos"  # Replace with your output directory
for url in urls:
    download_youtube_video(url, output_dir)
