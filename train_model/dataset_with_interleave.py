from audiomentations import (
    Compose, TimeStretch, PitchShift, AddGaussianNoise,
    AddGaussianSNR, AddColorNoise, AddBackgroundNoise
)
import numpy as np
from audiomentations import Compose, AddBackgroundNoise, TimeStretch, PitchShift, Shift
import librosa
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
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

def prepare_dataset(batch, processor, audio_column="audio", txt_column="transcription"): 
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
    audio_signals = resample_audio(audio_signals, sample_rates, target_sr=16000)
    
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
         
        print("Dataset preparation completed!") 
        return processed_dataset 
 
    except Exception as e: 
        print(f"Error in dataset preparation: {str(e)}") 
        raise

def apply_curriculum_augmentation(samples, phase):
    """Alternative version with more complex synthetic noise combinations."""
    
    no_augment = Compose([])

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
    
    # print(samples)

    audio = samples["audio"]["array"]
    
    if phase == 0:
        augmented_audio = no_augment(samples=audio, sample_rate=16000)
    elif phase == 1:
        augmented_audio = light_augment(samples=audio, sample_rate=16000)
    else:
        augmented_audio = strong_augment(samples=audio, sample_rate=16000)
    
    samples["audio"]["array"] = augmented_audio
    return samples

def augment_dataset_old(dataset, phase):
    print(f'[INFO] Applying augmentation for phase {phase}')
    return dataset.map(lambda x: apply_curriculum_augmentation(x, phase))

# Add these new functions
def create_curriculum_datasets_old(dataset, processor, phase_epochs_dict):
    """Create curriculum phases with increasing augmentation strength"""
    print("[INFO] Creating curriculum datasets...")
    
    def process_example(batch, augmentation_level):
        # Apply augmentation
        audio = augment_dataset(batch, augmentation_level)
        
        # Process features
        input_features = processor(
            audio["array"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features[0]

        # Process labels
        labels = processor.tokenizer(batch["transcription"]).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels
        }

    phases = []
    # 3 phases with different augmentation levels
    levels = [0,1,2] # phase_epochs_dict.keys()
    for level in levels:
        phase_ds = dataset.map(
            lambda x: process_example(x, level),
            remove_columns=dataset.column_names,
            num_proc=4
        )
        phases.append(phase_ds)
        print(f"[INFO] Created phase {level} with {len(phase_ds)} examples")
    
    return phases

def curriculum_interleave(datasets, epoch, total_epochs):
    """Dynamic dataset mixing weights based on training progress"""
    progress = epoch / total_epochs
    weights = [
        max(0, 1 - progress * 3),          # Phase 0 weight decreases
        max(0, min(1, 2 - progress * 3)),  # Phase 1 weight peaks mid-training
        max(0, progress * 3 - 1)           # Phase 2 weight increases
    ]
    total = sum(weights)
    return [w/total for w in weights]


def augment_dataset(example, phase):
    """Apply curriculum-based augmentation to a single example"""
    audio = example["audio"]
    
    # Initialize augmentation pipeline based on phase
    augmenter = get_augmentation_pipeline(phase)
    
    if augmenter:
        # Apply augmentations to audio array
        augmented_audio = augmenter(
            samples=audio["array"], 
            sample_rate=audio["sampling_rate"]
        )
        audio["array"] = augmented_audio
    
    return {"audio": audio}

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

def prepare_curriculum_datasets(dataset, processor, batch_size=32):
    """Create and cache curriculum phases"""
    print("Creating curriculum datasets...")
    
    # Create augmented datasets for each phase
    phases = []
    for level in [0, 1, 2]:
        print(f"Creating phase {level} dataset...")
        
        # Apply augmentation and processing
        phase_ds = dataset.map(
            lambda x: process_example_batch(x, processor, level),
            remove_columns=dataset.column_names,
            num_proc=os.cpu_count(),  # Use more processes for faster mapping
            batched=True,  # Process in batches for efficiency
            batch_size=100,  # Adjust based on memory constraints
            load_from_cache_file=False  # Disable caching to avoid conflicts
        )
        
        phases.append(phase_ds)
        print(f"Created phase {level} with {len(phase_ds)} examples")
    
    return phases

def process_example_batch(batch, processor, augmentation_level):
    """Process a batch of examples with specified augmentation level"""
    # Apply augmentation to audio
    augmented_audio = [
        augment_dataset({"audio": audio}, augmentation_level)["audio"]["array"]
        for audio in batch["audio"]
    ]
    
    # Process audio features in batch
    input_features = processor(
        augmented_audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_features
    
    # Process text labels in batch
    labels = processor.tokenizer(
        batch["transcription"],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_ids
    
    return {
        "input_features": input_features,
        "labels": labels
    }
    
def augment_dataset_batch(batch, phase):
    """Apply augmentation to a batch of audio samples"""
    augmenter = get_augmentation_pipeline(phase)
    if not augmenter:
        return batch
    
    # Apply augmentation to all samples in batch
    augmented_audio = [
        augmenter(samples=audio["array"], sample_rate=audio["sampling_rate"])
        for audio in batch["audio"]
    ]
    
    # Update audio arrays
    for i, audio in enumerate(batch["audio"]):
        audio["array"] = augmented_audio[i]
    
    return batch

def curriculum_interleave(datasets, epoch, total_epochs):
    """Dynamic dataset mixing weights based on training progress"""
    progress = epoch / total_epochs
    weights = [
        max(0, 1 - progress * 3),          # Phase 0 weight decreases
        max(0, min(1, 2 - progress * 3)),  # Phase 1 weight peaks mid-training
        max(0, progress * 3 - 1)           # Phase 2 weight increases
    ]
    total = sum(weights)
    return [w/total for w in weights]

    
def augment_dataset_batch(batch, phase):
    """Apply augmentation to a batch of audio samples"""
    augmenter = get_augmentation_pipeline(phase)
    if not augmenter:
        return batch
    
    # Apply augmentation to all samples in batch
    augmented_audio = [
        augmenter(samples=audio["array"], sample_rate=audio["sampling_rate"])
        for audio in batch["audio"]
    ]
    
    # Update audio arrays
    for i, audio in enumerate(batch["audio"]):
        audio["array"] = augmented_audio[i]
    
    return batch


def augment_dataset(dataset, phase_probabilites):
    print(f'[INFO] Applying augmentation using phase_probabilites: {phase_probabilites}')
    return dataset.map(lambda x: apply_curriculum_augmentation(x, phase_probabilites))