from dataclasses import dataclass
from typing import Any, List, Dict, Union
import torch
from multiprocessing import Value
from audiomentations import (
    Compose, TimeStretch, PitchShift, AddGaussianNoise,
    AddGaussianSNR, AddColorNoise, AddBackgroundNoise
)
import librosa
import numpy as np
import os

# ----------------------------------------------------------------- #
# ----------------------       Dataset       ---------------------- #
# ----------------------------------------------------------------- #

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingCurriculum:
    """ Used for dynamic curriculum learning with variable augmentation levels """
    processor: Any
    total_steps: int  # Total training steps
    current_step: Value  # Shared counter for tracking steps

    def __init__(self, processor, total_steps, probabilities_per_phase_dict):
        self.processor = processor
        self.total_steps = total_steps
        self.current_step = Value('i', 0)  # Shared integer counter
        self.probabilities_per_phase_dict = probabilities_per_phase_dict
        print(f'[INFO] DataCollator initialized with total_steps={total_steps}')
    
    def get_phase_probabilities(self):
        """Update phase probabilities dynamically based on the current step."""
        with self.current_step.get_lock():  # Ensure thread-safe access
            progress = self.current_step.value / self.total_steps
        
        if progress < 0.33:  # First third of training
            return self.probabilities_per_phase_dict["0"]  # Light augmentations
        
        elif progress < 0.66:  # Middle third of training
            return self.probabilities_per_phase_dict["2"] # Moderate augmentations
        else:  # Final third of training
            return self.probabilities_per_phase_dict["2"] # Heavier augmentations

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Update phase probabilities based on current step
        phase_probabilities = self.get_phase_probabilities()
        
        # Increment step counter (thread-safe)
        with self.current_step.get_lock():
            self.current_step.value += 1
        
        # Process and augment each feature in the batch
        processed_features = []
        for feature in features:
            # Extract audio and transcription
            audio = feature["audio"]
            transcription = feature["transcription"]
            
            # Normalize audio signal
            audio_signal = audio["array"]
            max_value = np.abs(audio_signal).max()
            if max_value > 1.0:
                audio_signal = audio_signal / 32768.0
            
            # Resample audio to 16kHz if necessary
            if audio["sampling_rate"] != 16000:
                audio_signal = librosa.resample(
                    audio_signal,
                    orig_sr=audio["sampling_rate"],
                    target_sr=16000
                )
            
            # Apply augmentation to raw audio
            augmented_audio = apply_curriculum_augmentation(audio_signal, 16000, phase_probabilities)
            
            # Process augmented audio into input features
            input_features = self.processor.feature_extractor(
                augmented_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=480000,  # Max audio length in samples (30 seconds @ 16kHz)
                truncation=True
            ).input_features[0]
            
            # Process text labels
            labels = self.processor.tokenizer(
                transcription,
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True
            ).input_ids[0]
            
            # Add processed features to the batch
            processed_features.append({
                "input_features": input_features,
                "labels": labels
            })
        
        # Proceed with padding and other processing steps as usual
        input_features = [{"input_features": feature["input_features"]} for feature in processed_features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in processed_features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
   
    
def resample_audio(audio_signals, sample_rates, target_sr=16000):
    """Resample audio signals to target sampling rate."""
    resampled_signals = []
    for signal, sr in zip(audio_signals, sample_rates):
        if sr != target_sr:
            # Resample using scipy
            resampled = librosa.resample(y=signal, orig_sr=sr, target_sr=target_sr)
            resampled_signals.append(resampled)
        else:
            resampled_signals.append(signal)
    return resampled_signals

def prepare_dataset(batch, processor, audio_column="audio", txt_column="transcription", target_sr=16000): 
    """Prepare dataset with proper padding and handling of variable-length inputs.""" 
    # Process audio 
    audios = batch[audio_column]
    
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
    audio_signals = resample_audio(audio_signals, sample_rates, target_sr=target_sr)
    
    # Process the first audio signal (since we're processing batch by batch)
    input_features = processor.feature_extractor( 
        audio_signals[0],  # Use the processed audio signal
        sampling_rate=16000,  # We know it's 16kHz after resampling
        return_tensors="pt", 
        padding="max_length", 
        max_length=480000,  # Max audio length in samples (30 seconds @ 16kHz) 
        truncation=True 
    ).input_features 
 
    # Process text 
    labels = processor.tokenizer( 
        batch[txt_column],  
        return_tensors="pt", 
        padding="max_length", 
        max_length=128, 
        truncation=True 
    ).input_ids 
 
    batch["input_features"] = input_features[0] 
    batch["labels"] = labels[0] 
    return batch 
     
def prepare_datasets(dataset, processor): 
    """Load and prepare all datasets with proper error handling.""" 
    try: 
        # Apply preprocessing 
        print("Applying preprocessing...") 
        processed_dataset = {} 
        for split in ["train", "test"]: 
            processed_dataset[split] = dataset[split].map( 
                lambda x: prepare_dataset(x, processor), 
                remove_columns=dataset[split].column_names, 
                desc=f"Processing {split} split", 
                # num_proc=4, 
            ) 
        
        print(processed_dataset['train'][0])
        print("Dataset preparation completed!") 
        return processed_dataset 
 
    except Exception as e: 
        print(f"Error in dataset preparation: {str(e)}") 
        raise
    
def apply_curriculum_augmentation(audio, sample_rate, phase_probabilities):
    """Applies augmentation using complex synthetic noise combinations depending on the sampled phase."""
    
    # audio = samples["audio"]["array"]
    # sample curriculum phase
    phase = np.random.choice([0, 1, 2], p=phase_probabilities)
    # get augmentation pipeline
    augmenter = get_augmentation_pipeline(phase)
    # apply augmentation
    augmented_audio = augmenter(samples=audio, sample_rate=sample_rate)
    # # update samples
    # samples["audio"]["array"] = augmented_audio
    
    return augmented_audio

def get_augmentation_pipeline(phase):
    """Get appropriate augmentation pipeline for curriculum phase"""
    
    if phase == 0:
        no_augment = Compose([])
        return no_augment
    
    elif phase == 1:
        light_augment = Compose([
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
            PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
            # Mix of different noise types
            AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.005,
                p=0.3
            ),
            AddColorNoise(
                # pink noise
                min_f_decay=-3.01,
                max_f_decay=-1.01,
                p=0.2
            )
        ])
        return light_augment
    elif phase == 2:
        # More complex noise patterns for advanced augmentation
        strong_augment = Compose([
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
            
            # Layer 1: Base environmental noise simulation
            AddColorNoise(
                # brown noise, Good for environmental noise
                min_f_decay=-6.02,
                max_f_decay=2.01,
                p=0.4
            ),
            
            # Layer 2: Mid-frequency noise
            AddColorNoise(
                # pink noise
                min_f_decay=-3.01,
                max_f_decay=-1.01,
                p=0.3
            ),
            
            # Layer 3: High-frequency detail
            AddGaussianNoise(
                p=0.3
            ),
            
            # Layer 4: SNR-controlled noise
            AddGaussianSNR(
                min_snr_db=10,
                max_snr_db=20,
                p=0.3
            )
        ])
        return strong_augment