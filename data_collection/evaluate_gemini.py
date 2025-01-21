import os
import google.generativeai as genai
from jiwer import wer, cer
import pandas as pd
from datasets import (
    load_dataset,
    Dataset,
)
from dotenv import load_dotenv
import soundfile as sf
import time
from google.api_core import exceptions
import random
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

HF_TOKEN = os.environ["HF_TOKEN"]

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def transcribe_with_gemini(audio_path, prompt="You are an expert transcriber specializing in Moroccan Darija (الدارجة المغربية) with a meticulous eye for code-switching. Your sole task is to provide verbatim transcriptions of spoken Moroccan Darija from audio file. Crucially, you must accurately capture and represent all instances of code-switching with French and English words.\n\nCode-Switching Rules:\n\nWhen a speaker switches to French or English, transcribe those specific words or phrases in their original Latin script (lowercase). There can be many in the same input (happens frequently in some audios).\n\nImmediately following the Latin script code-switched portion, resume transcribing in Arabic script (الحروف العربية) for the remaining Darija.\n\nDo not capitalize code-switched words within a sentence.\n\nDo not attempt to translate or interpret any part of the audio. Your role is purely transcription.\n\nInput:\n\n Your output will be based on the content of the spoken Darija in the audio. You are not responsible for processing the audio file directly.\n\nOutput:\n\nprovide a single line containing the complete verbatim transcription of the associated spoken Darija, adhering to all code-switching rules.\n\nImportant Considerations:\n\nAccuracy is paramount: Precisely represent the speaker's words, including any fillers or hesitations.\n\nFidelity to Code-Switching: Your ability to flawlessly switch between Arabic and Latin scripts at the exact point of code-switching is essential.\n\nNo Interpretation: Only transcribe; avoid explanations or translations.\n\nExample Output (Illustrative):\n\nكانت واحد la soirée زوينة بزاف\n\nهذاك le problème كبير بزاف\n\nدابا غادي ندير un break صغير و نرجع\n\nExample Demonstrating Handling of a long code-switched section\n\nقلت ليه خاصك to be more proactive باش توصل لهدف ديالك\n\nBy following these guidelines precisely, you will create accurate and usable transcripts reflecting the nuances of Moroccan Darija speech with seamless code-switching."):
    """Transcribe an audio file using Gemini and return the transcription."""
    success = False
    attempt = 0
    max_retries = 100
    other_exceptions_max_retries = 3
    base_delay = 2  # Base delay in seconds
    while not success and attempt < max_retries:
        try :
            file = upload_to_gemini(audio_path, mime_type="audio/mpeg")
            
            # Configure model
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=generation_config,
            )
            
            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [file],
                    },
                ]
            )
        
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
                    print(f"Skipping chunk {audio_path} due to finish reason: {first_candidate.finish_reason} and safety_ratings: {first_candidate.safety_ratings if hasattr(first_candidate, 'safety_ratings') else 'N/A'}")
                    success = True  # Skip this chunk
                    break
                else:
                    # Process successful response
                    transcription = response.text
                    return transcription
            else:
                print(f"Skipping chunk {audio_path} due to no candidates in the response")
                success = True
                break
        
        except exceptions.ResourceExhausted as e:
            attempt += 1
            if attempt == max_retries:
                print(f"Failed to transcribe {audio_path} after {max_retries} attempts")
                raise

            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Resource exhausted, attempt {attempt}/{max_retries}. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            attempt += 1
            if attempt == other_exceptions_max_retries:
                success = True
                break
    return ""

def calculate_metrics_gemini(dataset, model_name="gemini-2.0-flash-exp", audio_folder="audios"):
    """Calculate transcription metrics (WER and CER) using Gemini."""
    
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
        print(f"[INFO] Folder '{audio_folder}' created to store audio files.")
    else:
        print(f"[INFO] Folder '{audio_folder}' already exists.")
        
    predictions = []
    references = dataset['transcription']
    
    for i, audio_entry in enumerate(dataset['audio']):
        # Extraire le tableau numpy et le taux d'échantillonnage
        audio_array = audio_entry["array"]
        sampling_rate = audio_entry["sampling_rate"]
        
        # Construire un chemin de fichier unique pour chaque fichier audio
        audio_file_path = os.path.join(audio_folder, f"audio_{i}.wav")
        
        # Sauvegarder l'audio dans le dossier spécifié
        sf.write(audio_file_path, audio_array, sampling_rate)
        print(f"[INFO] Saved audio file: {audio_file_path}")
        
        # Transcrire l'audio avec Gemini
        transcription = transcribe_with_gemini(audio_file_path)
        predictions.append(transcription)
        
        print(f"[INFO] Transcription: {transcription}")
    
    # Calculer WER et CER
    wer_score = wer(references, predictions)
    cer_score = cer(references, predictions)
    
    metrics = {
        'model': model_name,
        'wer': wer_score,
        'cer': cer_score,
        'timestamp': pd.Timestamp.now()  # Ajout d'un timestamp pour le suivi
    }
    
    print(f"[INFO] Metrics for {model_name}: WER={wer_score:.4f}, CER={cer_score:.4f}")
    return metrics

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Eval", split="validation", token=HF_TOKEN)
    
    # Calculate metrics for Gemini
    gemini_metrics = calculate_metrics_gemini(dataset)
    
    print("[INFO] Final Gemini Metrics:")
    print(gemini_metrics)

    # Push to Hub paths
    EVAL_DATA_HUB = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Evaluated"
    DATA_HUB = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Evaluated"
    METRICS_HUB = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Metrics"
    
    COMMIT_MSG_TRANSCRIPTIONS = "Added transcriptions for model: Gemini 2.0 Flash"
    COMMIT_MSG_METRICS = "Added WER and CER metrics for model: Gemini 2.0 Flash"
    
     # Save transcriptions
    dataset.push_to_hub(DATA_HUB, commit_message=COMMIT_MSG_TRANSCRIPTIONS, token=HF_TOKEN)
    
    # Create and save metrics dataset
    metrics_dataset = Dataset.from_pandas(pd.DataFrame(gemini_metrics))
    metrics_dataset.push_to_hub(METRICS_HUB, commit_message=COMMIT_MSG_METRICS, token=HF_TOKEN)
    
    print('[INFO] Results and metrics saved successfully.')