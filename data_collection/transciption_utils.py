import json
import time
import random
from pydub import AudioSegment
import google.generativeai as genai
from tqdm import tqdm
from google.api_core import exceptions
from constants import *

# Upload audio to Gemini with retries
def upload_to_gemini(path, mime_type=None, MAX_RETRIES_GEMINI_TRANSCRIBE=3, base_delay=1):
    for attempt in range(MAX_RETRIES_GEMINI_TRANSCRIBE):
        try:
            file = genai.upload_file(path, mime_type=mime_type)
            return file
        except exceptions.ResourceExhausted:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

# Split audio into chunks
def split_audio(audio_path, chunk_duration=10):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration * 1000):
        chunk = audio[i:i + chunk_duration * 1000]
        chunk_path = f"{chunks_folder}/chunk_{i // (chunk_duration * 1000)}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    return chunks

def transcribe_audio(audio_chunks, model, channel_url, video_url, title, prompt="Here is the audio:"):
    """Transcribes a list of audio chunks using the Gemini model with robust error handling."""
    transcriptions = []
    transcripts_path = f"{transcripts_folder}/transcripts.json"
    counter = 0
    base_delay = 2  # Base delay in seconds

    for chunk_path in tqdm(audio_chunks):
        success = False
        attempt = 0

        while not success and attempt < MAX_RETRIES_GEMINI_TRANSCRIBE:
            try:
                # Rate limiting check
                if counter % 25 == 0 and counter != 0:
                    print("[INFO] Rate limit prevention pause...")
                    time.sleep(60)  # Standard cool-down period

                # Upload file with its own retry mechanism
                file = upload_to_gemini(chunk_path, mime_type="audio/mpeg")

                # Start chat session and send message
                chat_session = model.start_chat(history=[{"role": "user", "parts": [file]}])
                response = chat_session.send_message(prompt)

                # Check if there are any candidates and get the first one
                if response.candidates:
                    first_candidate = response.candidates[0]
                    
                    # Check finish_reason
                    finish_reason = getattr(first_candidate, 'finish_reason', None)
                    
                    if finish_reason is not None and (
                         (isinstance(finish_reason, str) and finish_reason.upper() != "STOP") or
                         (isinstance(finish_reason, int) and finish_reason != 1)  #STOP enum value is 1 as an integer
                        ):
                        # print(f"[INFO] response: {response}")
                        # print(f"[INFO] first_candidate: {first_candidate}")
                        # print(f"[INFO] first_candidate.finish_reason: {first_candidate.finish_reason}")
                        # print(f"[INFO] Skipping chunk {chunk_path} due to finish reason: {first_candidate.finish_reason} and safety_ratings: {first_candidate.safety_ratings if hasattr(first_candidate, 'safety_ratings') else 'N/A'}")
                        print(f"[INFO] Skipping chunk {chunk_path} due to finish reason: {first_candidate.finish_reason}...")
                        success = True  # Skip this chunk
                        break
                    else:
                        # Process successful response
                        transcription = response.text
                        transcriptions.append({
                            "audio_path": chunk_path,
                            "transcription": transcription,
                            "channel_url": channel_url,
                            "video_url": video_url,
                            "title": title,
                            "attempt": attempt + 1
                        })
                else:
                     print(f"[INFO] Skipping chunk {chunk_path} due to no candidates in the response")
                     success = True
                     break

                # Save progress after each successful transcription
                with open(transcripts_path, "w", encoding="utf-8") as f:
                    json.dump(transcriptions, f, indent=4, ensure_ascii=False)

                success = True
                counter += 1
                # print(f"[INFO] Successfully transcribed chunk {len(transcriptions)}/{len(audio_chunks)}")

            except exceptions.ResourceExhausted as e:
                attempt += 1
                if attempt == MAX_RETRIES_GEMINI_TRANSCRIBE:
                    print(f"[INFO] Failed to transcribe {chunk_path} after {MAX_RETRIES_GEMINI_TRANSCRIBE} attempts")
                    raise

                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[INFO] Resource exhausted, attempt {attempt}/{MAX_RETRIES_GEMINI_TRANSCRIBE}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

            except Exception as e:
                error = str(e)
                if 'finish_reason: SAFETY' in error:
                    #to avoid exponential waiting time due to safety, it will almost never go through so just skip it
                    print(f"[INFO] Skipping chunk {chunk_path} due to finish reason: SAFETY...")
                    success = True  # Skip this chunk
                    break
                else:
                    print(f"[INFO] Unexpected error: {error}")
                    attempt += 1
                    if attempt == MAX_RETRIES_GEMINI_TRANSCRIBE:
                        raise
                    time.sleep(min(MAX_WAIT_TIME_SECONDS, base_delay * (2 ** attempt)))

    return transcriptions
