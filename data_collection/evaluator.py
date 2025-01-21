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
)
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc
import torch
from jiwer import wer, cer
import pandas as pd

import os
# if you have access to a GPU, otherwise comment this line. If you have a single GPU, you can set it to "0". If you have multiple GPUs, you can set it to "0,1" for example.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----------------------------------------------------------------- #
# ------------------- Load Transcription Model -------------------- #
# ----------------------------------------------------------------- #

def load_pipeline(model_id, force_ar=False):
    """ Create a pipeline for automatic speech recognition using Hugging Face's Transformers library."""
    if force_ar:
        print("[INFO] Loading processor and model...")
        processor = WhisperProcessor.from_pretrained(model_id)
        # Load config and enable Flash Attention 2
        config = WhisperConfig.from_pretrained(model_id)
        config.use_flash_attention_2 = True
        
        # Load model with modified config
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            config=config,
        )
        
        # Force Moroccan Arabic (Darija) output
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids
        model.config.suppress_tokens = []
        
        # Ensure the model always uses Moroccan Arabic language token
        model.config.forced_language_ids = processor.tokenizer.get_vocab()["<|ar|>"]
        
        # Define generation config for consistent Moroccan Arabic output
        model.generation_config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.suppress_tokens = []
        model.generation_config.forced_language_ids = processor.tokenizer.get_vocab()["<|ar|>"]
        
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor
        )
        
    return pipeline("automatic-speech-recognition", model=model_id)

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
    results = transcriptor(audio_signals, batch_size=batch_size)

    # Extract the transcriptions
    transcriptions = [result["text"] for result in results]
    
    if force_ar:
        return {f'{model_name+'-forced-ar'}': transcriptions}
    else:
        return {f'{model_name}': transcriptions}
        

# ----------------------------------------------------------------- #
# ---------------------- Evaluation Metrics ---------------------- #
# ----------------------------------------------------------------- #

def calculate_metrics(predictions, references, model_name):
    """Calculate WER and CER for a batch of predictions."""
    wer_score = wer(references, predictions)
    cer_score = cer(references, predictions)
    return {
        'model': model_name,
        'wer': wer_score,
        'cer': cer_score,
        'timestamp': pd.Timestamp.now()  # Add timestamp for tracking
    }

    
if __name__ == "__main__":
    
    # Push to Hub paths
    EVAL_DATA_HUB = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Evaluated"
    DATA_HUB = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Evaluated"
    METRICS_HUB = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Metrics"
    
    FORCE_AR = False  # Force Moroccan Arabic output for whisper v3 models
    
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
        # "OmarAladdin/speecht5_finetuned_Egyptian_ASR_MGB_3": 16,
        "mukhtar/whisper-V3-MGB3-3EP" : 16,
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
    
    COMMIT_MSG_TRANSCRIPTIONS = "Added transcriptions for models:"
    COMMIT_MSG_METRICS = "Added WER and CER metrics for models:"
    for model_name in MODEL_PATHS_AND_BATCH_SIZE_DICT.keys():
        COMMIT_MSG_TRANSCRIPTIONS += f" {model_name}, "
        COMMIT_MSG_METRICS += f" {model_name}, "
    COMMIT_MSG_TRANSCRIPTIONS = COMMIT_MSG_TRANSCRIPTIONS[:-2] + "."
    COMMIT_MSG_METRICS = COMMIT_MSG_METRICS[:-2] + "."
    
    # Load Eval Dataset
    dataset = load_dataset("BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Eval", split="validation")
    
    print(f"[INFO] Dataset: {dataset}")
    
    for model_name, batch_size in tqdm(MODEL_PATHS_AND_BATCH_SIZE_DICT.items(), desc="Processing Models"): 
        print(f"[INFO] Transcribing using model: {model_name}...")
        transcriber = load_pipeline(model_name, force_ar=FORCE_AR)

        # Apply transcription with batching
        new_transcriptions = dataset.map(
            lambda examples: batch_transcription(examples['audio'], transcriber, model_name, batch_size, force_ar=FORCE_AR),
            batched=True,
            batch_size=batch_size,
            desc="Transcribing...",
        )
        
        # Add the new transcriptions to the existing dataset
        dataset = dataset.add_column(f'{model_name}', new_transcriptions[f'{model_name}'])
        
        # Calculate metrics
        if FORCE_AR:
            model_name = f'{model_name}-forced-ar'
            
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