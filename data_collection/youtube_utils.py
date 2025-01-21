import soundfile as sf
import datasets
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    Audio,
)
import pandas as pd
import yt_dlp
import os
import json
from tqdm import tqdm
from transciption_utils import (
    split_audio, 
    transcribe_audio
)
import time
import random
from datetime import datetime
from constants import *

import yt_dlp
import os
import time
import json
from pathlib import Path

def download_video(video_url, download_path, output_filename="audio"):
    """
    Simplified download function focusing on format selection and cookie handling.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Try to find or create a cookies directory
    cookies_dir = Path.home() / '.yt-dlp' / 'cookies'
    cookies_dir.mkdir(parents=True, exist_ok=True)
    cookies_file = cookies_dir / 'youtube.com_cookies.txt'

    # Basic options focusing on reliable format selection
    ydl_opts = {
        'format': 'ba',  # Best audio only
        'outtmpl': os.path.join(download_path, f"{output_filename}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',  # Lower quality might help
        }],
        
        # Minimal retry settings
        'retries': 3,
        'fragment_retries': 3,
        'skip_unavailable_fragments': True,
        
        # Network settings
        'socket_timeout': 15,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
    }

    # Try different format selections
    format_options = ['ba', 'bestaudio[ext=m4a]', 'bestaudio[ext=webm]', 'worstaudio']
    
    for format_option in format_options:
        try:
            print(f"\nTrying format: {format_option}")
            ydl_opts['format'] = format_option
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First try to just get video info
                try:
                    info = ydl.extract_info(video_url, download=False)
                    print(f"Available formats: {[f['format_id'] for f in info['formats']]}")
                except Exception as e:
                    print(f"Info extraction failed: {e}")
                    continue

                # Then try the actual download
                info = ydl.extract_info(video_url, download=True)
                filename = ydl.prepare_filename(info)
                
                # If we got here, download was successful
                print(f"Successfully downloaded with format {format_option}")
                return True

        except Exception as e:
            print(f"Failed with format {format_option}: {str(e)}")
            time.sleep(5)
            continue

    print("All format options failed")
    return False
    
# Recursive function to handle nested entries and extract video URLs
def extract_video_urls(entries):
    video_urls = []
    for entry in entries:
        if entry and 'url' in entry:
            video_id = entry.get('id')
            if video_id:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_urls.append(video_url)
            else:
                video_urls.append(entry['url'])

        # If the entry has nested entries, extract them recursively
        if "entries" in entry:
            video_urls.extend(extract_video_urls(entry["entries"]))
    
    return video_urls

# Fetch and filter Creative Commons videos from a YouTube channel
def fetch_cc_by_videos(channel_url, MAX_VIDEOS):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,  # Extract metadata without downloading
        'skip_download': True,
        'force_generic_extractor': True,  # Force generic extraction if needed
        'ignoreerrors': True,  # Skip videos with errors
        'playlistend': 1000,  # Limit to first 1000 videos to handle pagination
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
        except Exception as e:
            print(f"[ERROR] Failed to extract info: {e}")
            return []

        # Save info for debugging
        with open('info.json', 'w') as f:
            json.dump(info, f, indent=4)

        # Check if 'entries' exist in info
        entries = info.get('entries', [])
        if not entries:
            print("[WARNING] No videos found in 'entries'.")
            return []

        # Extract video URLs recursively from entries
        video_urls = extract_video_urls(entries)
        # video_urls = video_urls[1:]

        print(f"[INFO] Found {len(video_urls)} video URLs.")

        # Optional: Filter for Creative Commons (CC-BY) videos
        cc_by_video_urls = []
        cpt = 0
        for video_url in tqdm(video_urls):
            if cpt == MAX_VIDEOS:
                break
            try:
                video_info = ydl.extract_info(video_url, download=False)
                license_type = video_info.get('license', '')
                if "Creative Commons" in license_type:
                    # print(f"[✔] CC-BY Video: {video_url}")
                    cc_by_video_urls.append(video_url)
            except Exception as e:
                print(f"[ERROR] Failed to check license for {video_url}: {e}")

            cpt = cpt + 1
        print(f"[INFO] Found {len(cc_by_video_urls)} CC-BY videos.")
        return cc_by_video_urls


# Process all videos from specified channels
def process_channels(channel_urls, model, download_path, MAX_VIDEOS, output_filename="audio", saving_frequency=24, IS_FIRST_TIME=False):
    
        
    for channel_url in channel_urls:
        print('-' * 50)
        print(f"[INFO] Processing channel: {channel_url}")
        video_urls = fetch_cc_by_videos(channel_url, MAX_VIDEOS)

        video_index = 0
        for video_url in tqdm(video_urls):
            
            # download the video in MP3
            download_video(video_url, download_path)
            video_info = yt_dlp.YoutubeDL().extract_info(video_url, download=False)
            title = video_info.get('title', 'Unknown Title')
            print(f"[INFO] Processing video: {title}")
            
            # Assuming the video has an associated audio file that has been downloaded
            audio_path = f"{download_path}/{output_filename}.mp3"
            
            if os.path.exists(audio_path):
                print(f"[INFO] Splitting audio for: {title}")
                audio_chunks = split_audio(audio_path)
                print(f"[INFO] Transcribing audio for: {title}")
                
                # Splitting audio_chunks into batches based on saving_frequency
                batch_index = 0
                total_batches = (len(audio_chunks) + saving_frequency - 1) // saving_frequency  # Total batches
                
                for i in range(0, len(audio_chunks), saving_frequency):
                    batch = audio_chunks[i:i + saving_frequency]
                    transcriptions = transcribe_audio(batch, model, channel_url, video_url, title)
                    
                    if IS_FIRST_TIME:
                        # Initialize the Hugging Face dataset (start with an empty dataset)
                        dataset_columns = {
                            "audio": datasets.Audio(),
                            "transcription": datasets.Value("string"),
                            "channel_url": datasets.Value("string"),
                            "video_url": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "attempt": datasets.Value("int32")
                        }
                        # Initialize an empty dataset
                        dataset = Dataset.from_dict({col: [] for col in dataset_columns})
                        IS_FIRST_TIME = False
                    else:
                        time.sleep(10)
                        dataset = load_dataset(HF_DATA_PATH, split="train")

                    # Append data to the Hugging Face dataset
                    for transcription in transcriptions:
                        row = {
                            "audio": transcription["audio_path"],  # Audio path will be cast to Audio() later
                            "transcription": transcription["transcription"],
                            "channel_url": transcription["channel_url"],
                            "video_url": transcription["video_url"],
                            "title": transcription["title"],
                            "attempt": transcription["attempt"],
                            "video_index": f'{video_index}/{len(video_urls)}'
                        }
                        dataset = dataset.add_item(row)
                    
                    # Ensure the audio column is properly recognized as Audio type
                    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

                    # Push each batch to Hugging Face Hub
                    dataset.push_to_hub(
                        HF_DATA_PATH,
                        commit_message=f"Added transcripts and audios for channel {channel_url}, batch_index={batch_index + 1}/{total_batches}",
                        # token=TOKEN
                    )
                    print(f"[INFO] Pushed batch {batch_index + 1}/{total_batches} to Hugging Face hub.")
                    batch_index += 1

                print(f"Transcriptions added in dataset for: {title}")
                
                # Delete the audio file to save disk space
                os.remove(audio_path)
                print(f"[INFO] Deleted audio file: {audio_path}")
            else:
                print(f"[WARNING] Audio file not found for: {title}")
                
            video_index = video_index + 1
