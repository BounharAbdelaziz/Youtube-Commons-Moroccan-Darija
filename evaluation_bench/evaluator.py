from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
)
from transformers import (
    pipeline,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor,
)
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc
import torch
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces
import re
import unicodedata
import pandas as pd

import os
# if you have access to a GPU, otherwise comment this line. If you have a single GPU, you can set it to "0". If you have multiple GPUs, you can set it to "0,1" for example.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
import json
from nemo.collections.asr.metrics.wer import word_error_rate
from tqdm import tqdm
import re

# ----------------------------------------------------------------- #
# ------------------- Load Transcription Model -------------------- #
# ----------------------------------------------------------------- #

def load_pipeline(model_id, batch_size=16):
    """ Create a pipeline for automatic speech recognition using Hugging Face's Transformers library."""
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print("[INFO] Loading processor and model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
    # model.config.forced_decoder_ids = forced_decoder_ids
    
    # # Ensure suppress_tokens is not empty
    # if model.config.suppress_tokens is None or len(model.config.suppress_tokens) == 0:
    #     model.config.suppress_tokens = [processor.tokenizer.pad_token_id]
    
    # # Ensure the model always uses Moroccan Arabic language token
    # model.config.forced_language_ids = processor.tokenizer.get_vocab()["<|ar|>"]
    
    # # Define generation config for consistent Moroccan Arabic output
    # model.generation_config.forced_decoder_ids = forced_decoder_ids
    # model.generation_config.suppress_tokens = model.config.suppress_tokens
    # model.generation_config.forced_language_ids = processor.tokenizer.get_vocab()["<|ar|>"]
    
    # create a speech-recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=batch_size,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

# def load_pipeline(model_id, batch_size=16):
#     """ Create a pipeline for automatic speech recognition using Hugging Face's Transformers library."""
    
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
#     print("[INFO] Loading processor and model...")
#     model = AutoModelForSpeechSeq2Seq.from_pretrained(
#         model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
#     )
#     model.to(device)
#     processor = AutoProcessor.from_pretrained(model_id)
    
#     forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
#     model.config.forced_decoder_ids = forced_decoder_ids
#     model.config.suppress_tokens = []
    
#     # Ensure the model always uses Moroccan Arabic language token
#     model.config.forced_language_ids = processor.tokenizer.get_vocab()["<|ar|>"]
    
#     # Define generation config for consistent Moroccan Arabic output
#     model.generation_config.forced_decoder_ids = forced_decoder_ids
#     model.generation_config.suppress_tokens = []
#     model.generation_config.forced_language_ids = processor.tokenizer.get_vocab()["<|ar|>"]
    
#     # create a speech-recognition pipeline
#     pipe = pipeline(
#         "automatic-speech-recognition",
#         model=model,
#         tokenizer=processor.tokenizer,
#         feature_extractor=processor.feature_extractor,
#         max_new_tokens=128,
#         chunk_length_s=30,
#         batch_size=batch_size,
#         return_timestamps=False,
#         torch_dtype=torch_dtype,
#         device=device,
#     )
#     return pipe
    
# ----------------------------------------------------------------- #
# ---------------------- Audio sampling rate ---------------------- #
# ----------------------------------------------------------------- #

def resample_audio(audio_signals, sample_rates, target_sr=16000):
    """Resample audio signals in parallel."""
    def resample(signal, sr):
        if sr != target_sr:
            return librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        return signal

    with ThreadPoolExecutor() as executor:
        resampled_signals = list(executor.map(resample, audio_signals, sample_rates))
    return resampled_signals

# ----------------------------------------------------------------- #
# ---------------------- Batch Transcription ---------------------- #
# ----------------------------------------------------------------- #

def batch_transcription(audios, transcriptor, model_name, batch_size=4, force_ar=False):
    """Optimized batch transcription function for batch inputs with variable-length audio."""
    
    # Check if audios is a list or single dictionary
    if isinstance(audios, dict):
        audios = [audios]  # Convert to a list for consistency
    
    # Extract audio signals and sampling rates as lists (not numpy array)
    audio_signals = [audio['array'] for audio in audios]
    sample_rates = [audio['sampling_rate'] for audio in audios]

    # Normalize audio signals
    for i in range(len(audio_signals)):
        max_value = np.abs(audio_signals[i]).max()
        if max_value > 1.0:
            audio_signals[i] = audio_signals[i] / 32768.0

    # Resample all audio signals to 16kHz in parallel
    audio_signals = resample_audio(audio_signals, sample_rates, target_sr=16000)

    # Transcribe the batch
    if force_ar:
        # results = transcriptor(audio_signals, batch_size=batch_size, generate_kwargs = {"language":"<|ar|>","task": "transcribe"})
        results = transcriptor(audio_signals, batch_size=batch_size) #, generate_kwargs = {"language":"<|ar|>","task": "transcribe"})
    else:
        results = transcriptor(audio_signals, batch_size=batch_size, generate_kwargs = {"task": "transcribe"})
        
    # Extract the transcriptions
    if isinstance(results, dict):  # Handle single result case
        results = [results]
        
    transcriptions = [result["text"] for result in results]
    
    # if force_ar:
    #     return {f'{model_name+'-forced-ar'}': transcriptions}
    # else:
    return {f'{model_name}': transcriptions}
        

# ----------------------------------------------------------------- #
# ----------------------     Evaluation      ---------------------- #
# ----------------------------------------------------------------- #

def remove_diacritics(text):
    """Remove diacritics, Hamzas, and Maddas."""
    return ''.join(
        [char for char in unicodedata.normalize('NFD', text)
         if unicodedata.category(char) != 'Mn']
    )

def convert_eastern_to_western_arabic_numerals(text):
    """Convert Eastern Arabic numerals to Western Arabic numerals."""
    eastern_arabic = {'٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'}
    return ''.join([eastern_arabic.get(c, c) for c in text])

def custom_preprocess(text):
    """Custom preprocessing function for both reference and prediction."""
    # Step 1: Remove all punctuation except `%` and `@`
    text = re.sub(r'[^\w\s%@]', '', text)
    
    # Step 2: Remove diacritics, Hamzas, and Maddas
    text = remove_diacritics(text)
    
    # Step 3: Convert Eastern Arabic numerals to Western Arabic numerals
    text = convert_eastern_to_western_arabic_numerals(text)
    
    # Step 4: Return the processed text
    return text

def normalize_arabic_text(text):
    """
    Arabic text normalization:
    1. Remove punctuation
    2. Remove diacritics
    3. Eastern Arabic numerals to Western Arabic numerals

    Arguments
    ---------
    text: str
        text to normalize
    Output
    ---------
    normalized text
    """
    # Remove punctuation
    punctuation = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؛؟]'
    text = re.sub(punctuation, '', text)

    # Remove diacritics
    diacritics = r'[\u064B-\u0652]'  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, '', text)
    
    # Normalize Hamzas and Maddas
    text = re.sub('پ', 'ب', text)
    text = re.sub('ڤ', 'ف', text)
    text = re.sub(r'[آ]', 'ا', text)
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ء]', '', text)   

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    return text.strip()

def calculate_metrics(predictions, references, model_name):
    """
    Arguments
    ---------
    output_manifest: str
        path to the output manifest of the model inference

    Output
    ---------
    WER/CER
    """
    
    # transform = Compose([
    #     ToLowerCase(),
    #     custom_preprocess,
    #     RemoveMultipleSpaces(),
    # ])
    # references = [transform(r) for r in references]
    # predictions = [transform(p) for p in predictions]
    
    references = [normalize_arabic_text(r) for r in references]
    predictions = [normalize_arabic_text(p) for p in predictions]

    assert len(references) == len(predictions), "Predictions and references must have same length!"
    
    wer_percentage = word_error_rate(predictions, references) * 100
    cer_percentage = word_error_rate(predictions, references, use_cer=True) * 100
    
    return {
        'model': model_name,
        'wer': wer_percentage,
        'cer': cer_percentage,
        'timestamp': pd.Timestamp.now()  # Add timestamp for tracking
    }

    
if __name__ == "__main__":
    
    # Push to Hub paths
    # EVAL_DATA_HUB = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Eval"
    EVAL_DATA_HUB = "UBC-NLP/Casablanca"
    DATA_HUB = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Evaluated-v2"
    METRICS_HUB = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Metrics"
    
    if EVAL_DATA_HUB == "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Eval":
        eval_split = "validation"
        dataset = load_dataset(EVAL_DATA_HUB, split=eval_split)
    else:
        split = "test"
        target_language = "Morocco" # "Morocco" # :WER: 93.0429, CER: 49.4269 , "Egypt": WER: 56.6144, CER: 24.3552 # Mine BounharAbdelaziz/Morocco-Darija-STT-large-v1.4 WER: 68.0324, CER: 25.7769
         # Load Eval Dataset
        dataset = load_dataset(EVAL_DATA_HUB, target_language, split=split)
    
    print(f"[INFO] Dataset: {dataset}")
    
    # Force Arabic output for whisper models
    FORCE_AR = False #True  
    
    # Models paths
    MODEL_PATHS_AND_BATCH_SIZE_DICT = {
        # "BounharAbdelaziz/Morocco-Darija-STT-tiny"          : 32,  # requires 2GB of GPU memory
        # "BounharAbdelaziz/Morocco-Darija-STT-small"         : 32,  # requires 6GB of GPU memory
        # "BounharAbdelaziz/Morocco-Darija-STT-large-v1.2"    : 16,  # requires 16GB of GPU memory
        # "openai/whisper-large-v3-turbo"                     : 16,  # requires 12GB of GPU memory
        # "openai/whisper-large-v3"                           : 16,  # requires 18GB of GPU memory
        # "boumehdi/wav2vec2-large-xlsr-moroccan-darija"      : 32,  # requires 12GB of GPU memory
        # "abdelkader12/whisper-small-ar"                     : 32,  # requires 7GB of GPU memory
        # "ychafiqui/whisper-medium-darija"                   : 32,  # requires 14GB of GPU memory
        # "ychafiqui/whisper-small-darija"                    : 32,  # requires 8GB of GPU memory
        # "BounharAbdelaziz/Morocco-Darija-STT-tiny-v1.3"    : 32,  # requires 2GB of GPU memory
        # "BounharAbdelaziz/Morocco-Darija-STT-small-v1.3"    : 32,  # requires 6GB of GPU memory
        # "BounharAbdelaziz/Morocco-Darija-STT-large-turbo-v1.3"    : 32,  # requires 6GB of GPU memory
        # "OmarAladdin/speecht5_finetuned_Egyptian_ASR_MGB_3": 16, # NOOT WORKING
        # "mukhtar/whisper-V3-MGB3-3EP" : 16,
        # "smerchi/generated_whisper_test1" : 16,
        # "smerchi/generated_whisper_test2" : 16,
        # "facebook/seamless-m4t-v2-large" : 16,
        # "openai/whisper-large-v3"    : 32,  # requires 16GB of GPU memory
        # "openai/whisper-large-v2"    : 32,  # requires 16GB of GPU memory
        # "BounharAbdelaziz/Morocco-Darija-STT-large-v1.5"    : 32,  # requires 16GB of GPU memory
        # "BounharAbdelaziz/Morocco-Darija-STT-large-turbo-v1.5"    : 32,  # requires 6GB of GPU memory
        # "BounharAbdelaziz/Morocco-Darija-STT-small-v1.5"    : 32,  # requires 6GB of GPU memory
        "BounharAbdelaziz/Moroccan-Darija-STT-small-v1.6.4" : 128,                  # 
        # "BounharAbdelaziz/Moroccan-Darija-STT-large-v1.6.4" : 32,                  # 
        # "BounharAbdelaziz/Morocco-Darija-STT-large-v1.4" : 32,                  # WER: 85.5596, CER: 47.9522 on my eval with FALSE_AR=False and WER: 76.5343, CER: 30.9361 with FALSE_AR=True           || WER: 82.8748, CER: 44.8742 on Casablanca Eval with FALSE_AR=False and WER: 68.0659, CER: 25.7921 with FALSE_AR=True
        # "BounharAbdelaziz/Moroccan-Darija-STT-large-turbo-v1.5": 32,          # WER: WER: 92.1781, CER: 51.9015 on my eval with FALSE_AR=False                                                        || WER: 96.1870, CER: 57.1718 on Casablanca Eval with FALSE_AR=False and WER: 81.7292, CER: 35.9003 with FALSE_AR=True
        # # "BounharAbdelaziz/Moroccan-Darija-STT-large-turbo-v1.5.2": 32,      # WER: 93.7425, CER: 50.3169 on my eval with FALSE_AR=False                                                             || on Casablanca Eval with FALSE_AR=False
        # "openai/whisper-large-v3": 32,                                        # WER: 91.4561, CER: 49.1711 on my eval with FALSE_AR=False                                                             || WER: 94.1467, CER: 53.9172 on Casablanca Eval with FALSE_AR=False and WER: 92.7084, CER: 49.1501 with FALSE_AR=True
        # "BounharAbdelaziz/Moroccan-Darija-STT-large-v1.5.3"    : 32,          # WER: 125.7521, CER: 104.7538 on my eval with FALSE_AR=False                                                           || on Casablanca Eval with FALSE_AR=False
        # "BounharAbdelaziz/Moroccan-Darija-STT-large-v1.5.2"    : 32,          # WER: 103.6101, CER: 84.4954 on my eval with FALSE_AR=False                                                            || on Casablanca Eval with FALSE_AR=False
        # "BounharAbdelaziz/Moroccan-Darija-STT-large-v1.5.1"    : 32,          # WER: 126.1131, CER: 99.3418 on my eval with FALSE_AR=False                                                            || on Casablanca Eval with FALSE_AR=False
    }
                  
    # Load existing metrics if available
    try:
        existing_metrics = load_dataset(METRICS_HUB, split="train")
        metrics_data = existing_metrics.to_pandas().to_dict('records')
        print(f"[INFO] Loaded existing metrics with {len(metrics_data)} entries")
        # metrics_data = []
    except Exception as e:
        print(f"[INFO] No existing metrics found or error loading them: {e}")
        metrics_data = []
    
    # create commit messages
    COMMIT_MSG_TRANSCRIPTIONS = "Added transcriptions for models:"
    COMMIT_MSG_METRICS = "Added WER and CER metrics for models:"
    for model_name in MODEL_PATHS_AND_BATCH_SIZE_DICT.keys():
        COMMIT_MSG_TRANSCRIPTIONS += f" {model_name}, "
        COMMIT_MSG_METRICS += f" {model_name}, "
    COMMIT_MSG_TRANSCRIPTIONS = COMMIT_MSG_TRANSCRIPTIONS[:-2] + "."
    COMMIT_MSG_METRICS = COMMIT_MSG_METRICS[:-2] + "."
    
    
    for model_name, batch_size in tqdm(MODEL_PATHS_AND_BATCH_SIZE_DICT.items(), desc="Processing Models"): 
        print(f"[INFO] Transcribing using model: {model_name}...")
        transcriber = load_pipeline(model_name, batch_size=batch_size)

        # Apply transcription with batching
        new_transcriptions = dataset.map(
            lambda examples, idx: batch_transcription(examples['audio'], transcriber, model_name, batch_size, force_ar=FORCE_AR),
            batched=True,
            batch_size=batch_size,
            desc="Transcribing...",
            with_indices=True,  # Ensures alignment
        )
        
        # Add the new transcriptions to the existing dataset
        dataset = dataset.add_column(f'{model_name}', new_transcriptions[f'{model_name}'])
        
        # Calculate metrics
        # if FORCE_AR:
        #     model_name = f'{model_name}-forced-ar'
            
        model_predictions = dataset[f'{model_name}']
        ground_truth = dataset['transcription']
        
        metrics = calculate_metrics(model_predictions, ground_truth, model_name)
        metrics_data.append(metrics)
        
        print(f'[INFO] Finished transcribing with Model: {model_name}')
        print(f'WER: {metrics["wer"]:.4f}, CER: {metrics["cer"]:.4f}')
        
        # Free memory
        gc.collect()
        torch.cuda.empty_cache()
    
    print('[INFO] Done transcribing with all models...')
    print('[INFO] Saving results...')
    
    # Save transcriptions
    dataset.push_to_hub(DATA_HUB, commit_message=COMMIT_MSG_TRANSCRIPTIONS)
    
    # Create and save metrics dataset
    metrics_dataset = Dataset.from_pandas(pd.DataFrame(metrics_data))
    metrics_dataset.push_to_hub(METRICS_HUB, commit_message=COMMIT_MSG_METRICS)
    
    print('[INFO] Results and metrics saved successfully.')