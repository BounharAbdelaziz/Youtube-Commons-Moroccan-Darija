import os
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
from pydub import AudioSegment
import json
import time
from google.api_core import exceptions
import random

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

chunks_folder = "chunks"
transcripts_folder = "transcripts"
if not os.path.exists(chunks_folder):
    os.makedirs(chunks_folder)
if not os.path.exists(transcripts_folder):
    os.makedirs(transcripts_folder)

def upload_to_gemini(path, mime_type=None, max_retries=3, base_delay=1):
    """Uploads the given file to Gemini with retry logic."""
    for attempt in range(max_retries):
        try:
            file = genai.upload_file(path, mime_type=mime_type)
            print(f"Uploaded file '{file.display_name}' as: {file.uri}")
            return file
        except exceptions.ResourceExhausted as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Upload failed, retrying in {delay:.2f} seconds...")
            time.sleep(delay)

def split_audio(audio_path, chunk_duration=10):
    """Splits an audio file into 10-second chunks."""
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = chunk_duration * 1000
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"{chunks_folder}/chunk_{i // chunk_length_ms}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    return chunks

def transcribe_audio(audio_chunks, model, prompt):
    """Transcribes a list of audio chunks using the Gemini model with robust error handling."""
    transcriptions = []
    transcripts_path = "transcripts/transcripts.json"
    counter = 0
    max_retries = 1000
    base_delay = 2  # Base delay in seconds

    for chunk_path in tqdm(audio_chunks):
        success = False
        attempt = 0

        while not success and attempt < max_retries:
            try:
                # Rate limiting check
                if counter % 25 == 0 and counter != 0:
                    print("Rate limit prevention pause...")
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
                        print(f"response: {response}")
                        print(f"first_candidate: {first_candidate}")
                        print(f"first_candidate.finish_reason: {first_candidate.finish_reason}")
                        print(f"Skipping chunk {chunk_path} due to finish reason: {first_candidate.finish_reason} and safety_ratings: {first_candidate.safety_ratings if hasattr(first_candidate, 'safety_ratings') else 'N/A'}")
                        success = True  # Skip this chunk
                        break
                    else:
                        # Process successful response
                        transcription = response.text
                        transcriptions.append({
                            "audio_path": chunk_path,
                            "transcription": transcription,
                            "attempt": attempt + 1
                        })
                else:
                     print(f"Skipping chunk {chunk_path} due to no candidates in the response")
                     success = True
                     break

                # Save progress after each successful transcription
                with open(transcripts_path, "w", encoding="utf-8") as f:
                    json.dump(transcriptions, f, indent=4, ensure_ascii=False)

                success = True
                counter += 1
                print(f"Successfully transcribed chunk {len(transcriptions)}/{len(audio_chunks)}")

            except exceptions.ResourceExhausted as e:
                attempt += 1
                if attempt == max_retries:
                    print(f"Failed to transcribe {chunk_path} after {max_retries} attempts")
                    raise

                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Resource exhausted, attempt {attempt}/{max_retries}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                attempt += 1
                if attempt == max_retries:
                    raise
                time.sleep(base_delay * (2 ** attempt))

    return transcriptions

system_instruction="You are an expert transcriber specializing in Moroccan Darija (الدارجة المغربية) with a meticulous eye for code-switching. Your sole task is to provide verbatim transcriptions of spoken Moroccan Darija from audio files. Crucially, you must accurately capture and represent all instances of code-switching with French and English words.\n\nCode-Switching Rules:\n\nWhen a speaker switches to French or English, transcribe those specific words or phrases in their original Latin script (lowercase). There can be many in the same input (happens frequently in some audios).\n\nImmediately following the Latin script code-switched portion, resume transcribing in Arabic script (الحروف العربية) for the remaining Darija.\n\nDo not capitalize code-switched words within a sentence.\n\nDo not attempt to translate or interpret any part of the audio. Your role is purely transcription.\n\nInput:\n\nYou will be provided with audio file names (e.g., audio1.wav, audio2.mp3). Your output will be based on the content of the spoken Darija in the audio associated with that filename. You are not responsible for processing the audio files directly.\n\nOutput:\n\nFor each filename, provide a single line containing the complete verbatim transcription of the associated spoken Darija, adhering to all code-switching rules. Each transcription should be treated independently.\n\nImportant Considerations:\n\nAccuracy is paramount: Precisely represent the speaker's words, including any fillers or hesitations.\n\nFidelity to Code-Switching: Your ability to flawlessly switch between Arabic and Latin scripts at the exact point of code-switching is essential.\n\nNo Interpretation: Only transcribe; avoid explanations or translations.\n\nExample Output (Illustrative):\n\nكانت واحد la soirée زوينة بزاف\n\nهذاك le problème كبير بزاف\n\nدابا غادي ندير un break صغير و نرجع\n\nExample Demonstrating Handling of a long code-switched section\n\nقلت ليه خاصك to be more proactive باش توصل لهدف ديالك\n\nBy following these guidelines precisely, you will create accurate and usable transcripts reflecting the nuances of Moroccan Darija speech with seamless code-switching.",

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 120000,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=system_instruction
)

# File processing
# audio_file_path = "وجها لوجه - نقاش التداول - متداول أمام  د.محمد طلال لحلو.mp3"
audio_file_path = "Interview ＂MORO＂ ⧸ من المغرب إلى فرنسا.wav"

audio_chunks = split_audio(audio_file_path)
transcriptions = transcribe_audio(audio_chunks, model, prompt=system_instruction)

# Save the Dataset
dataset_file = "transcription_dataset.json"
with open(dataset_file, "w", encoding="utf-8") as f:
    json.dump(transcriptions, f, indent=4, ensure_ascii=False)

print(f"Transcription results saved to {dataset_file}")

# Clean up audio chunks
for chunk_path in audio_chunks:
    os.remove(chunk_path)