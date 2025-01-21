import os
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_utils import process_channels

from constants import *

if __name__ == "__main__":
    
    for folder in [downloads_folder, chunks_folder, transcripts_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    # Load environment variables
    load_dotenv()
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Gemini Configuration
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 120000,
            "response_mime_type": "text/plain",
        },
        system_instruction=SYSTEM_INSTRUCTION
    )
    
    process_channels(channel_urls, model, download_path=DOWNLOAD_PATH, MAX_VIDEOS=MAX_VIDEOS, IS_FIRST_TIME=IS_FIRST_TIME)
    print("Processing complete. Transcriptions saved.")
