import wandb
import os
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
)
from datasets import (
    load_dataset,
    Audio,
    DatasetDict,
    concatenate_datasets,
)
from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
)
import evaluate

import warnings

# ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def print_trainable_parameters(model):
    """ Prints the number of trainable parameters in the model and their names """
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        # else:
            # print(f"[INFO] Frozen parameter: {name}")
    print(
        f'[INFO] Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of all parameters)'
    )

def compute_metrics(pred, processor):
    """Compute metrics for evaluation."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    wer_percentage = wer * 100
    
    cer_metric = evaluate.load("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    cer_percentage = cer * 100
    return {"wer": wer_percentage, "cer": cer_percentage}

def prepare_dataset(batch, processor, audio_column="audio", txt_column="transcription"):
    """Prepare dataset with proper padding and handling of variable-length inputs."""
    # Process audio
    audio = batch[audio_column]
    input_features = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding="max_length",
        max_length=480000, # Max audio length in samples (30 seconds @ 16kHz)
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
        for split in ["train", "validation"]:
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

if __name__ == "__main__":
    
    model_size = "large"
    version = "v1.5"
    index_model = "3"
    project_name = f'{model_size}_{version}'
    
    freeze_feature_extractor_layer_names = [
        'model.encoder.conv1.',
        'model.encoder.conv2.',
        'model.encoder.embed_positions.',
        # 'model.encoder.layers.',
        # 'model.encoder.layer_norm.',
    ]
    
    FORCE_ARABIC_OUTPUTS = False
    FREEZE_FEATURE_EXTRACTOR = True
    
    # DATA_PATH = "BounharAbdelaziz/Mixed-Morocco-Darija-Amazigh-English-and-French-ASR"
    # DATA_PATH = "BounharAbdelaziz/Morocco-Darija-ASR-v1.2"
    TRAIN_DATA_PATH = "BounharAbdelaziz/Youtube-Commons-Moroccan-Darija-14h47"
    TEST_DATA_PATH = "BounharAbdelaziz/Moroccan-Darija-Youtube-Commons-Eval"
       
    BASE_MODELS = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "large": "openai/whisper-large-v3",
        "large-turbo": "openai/whisper-large-v3-turbo",
    }
    
    BATCH_SIZES = {
        "tiny": 32,
        "base": 32,
        "small": 32,
        "large-turbo": 32,
        "large": 16,
    }
    
    LEARNING_RATES = {
        "tiny": 2e-5,
        "base": 2e-5,
        "small": 3e-7,
        "large": 3e-7,
        "large-turbo": 3e-7,
    }
    
    N_EPOCHS_DICT = {
        "tiny": 60,
        "base": 60,
        "small": 60,
        "large": 60,
        "large-turbo": 60,
    }
    
    GRAD_ACC = {
        "tiny": 1,                  # 64 effective batch size
        "base": 1,                  # 256 effective batch size
        "small": 1,             # 1024 effective batch size
        "large-turbo": 1,       # 1024 effective batch size
        "large": 16,            # 1024 effective batch size
    }
    
    # Add gradient clipping settings
    MAX_GRAD_NORM = {
        "tiny": 0.9,
        "base": 0.9,
        "small": 0.9,
        "large": 0.9,  # Lower for larger models
        "large-turbo": 0.9,  # Lower for larger models
    }
    
    # Add gradient clipping settings
    WEIGHT_DECAYS = {
        "tiny": 0.0,
        "base": 0.0,
        "small": 0.0001,
        "large": 0.0005,  
        "large-turbo": 0.0005, 
    }
    
    BASE_MODEL = BASE_MODELS[model_size]
    
    PUSH_FINETUNED_MODEL_TO = f"BounharAbdelaziz/Moroccan-Darija-STT-{model_size}-{version}.{index_model}"
    
    OUTPUT_DIR = f"./Mixed/{BASE_MODEL}/Morocco-Darija-STT-{model_size}-{version}.{index_model}"
    OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Apply chmod -R 755 to OUTPUT_DIR
    print(f"[INFO] Applying chmod -R 755 to OUTPUT_DIR: {OUTPUT_DIR}...")
    os.system(f"chmod -R 755 {OUTPUT_DIR}")
    
    
    # DEBUG = True
    DEBUG = False
    
    # hyperparameters
    n_epochs = N_EPOCHS_DICT[model_size]
    lr = LEARNING_RATES[model_size]
    batch_size = BATCH_SIZES[model_size]
    weight_decay = WEIGHT_DECAYS[model_size]
    gradient_accumulation_steps = GRAD_ACC[model_size]
    max_grad_norm = MAX_GRAD_NORM[model_size]  # Get max gradient norm for current model size
    eval_strategy='steps'
    save_strategy='steps'
    warmup_ratio = 0.1 # 10% of total steps
    logging_steps = 5
    eval_steps = 25
    save_steps = 25
    lr_scheduler_type = "cosine"
    run_name = f"{model_size}-{version}-ep-{n_epochs}-bs-{batch_size}-g_acc-{gradient_accumulation_steps}-wd-{weight_decay}"

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged, all runs will be under this project
        project=project_name,   
        # Group runs by model size
        group=model_size,       
        # Unique run name
        name=run_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "weight_decay": weight_decay,
            "dataset": TRAIN_DATA_PATH,
        }
    )


    print(f"[INFO] Finetuning base model: {BASE_MODEL}...")
    print(f"[INFO] Using training dataset: {TRAIN_DATA_PATH}...")
    print(f"[INFO] Using test dataset: {TEST_DATA_PATH}...")
    print(f"[INFO] Finetuned model will be pushed to hub: {PUSH_FINETUNED_MODEL_TO}...")

    # Training parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        load_best_model_at_end=True,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,  # Linear learning rate scheduler
        warmup_ratio=warmup_ratio,  # Warmup for the first steps
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=3,
        num_train_epochs=n_epochs,
        predict_with_generate=True,
        # push_to_hub=True,
        # logging_dir="./logs",
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,  # Add gradient clipping
        fp16=False,
        bf16=True, # Mixed precision training
        # bf16_full_eval=True,  # Use BF16 for evaluation too
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        gradient_checkpointing=True,  # Enable gradient checkpointing for additional memory savings
        logging_steps=logging_steps,
        report_to="wandb",  # Enable WandB
        run_name=run_name,
        # Enable pushing to Hub after every checkpoint
        push_to_hub=True,  
        hub_model_id=PUSH_FINETUNED_MODEL_TO,
        hub_strategy="every_save",  # Push model to hub after every save
    )

    print("[INFO] Loading processor and model...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    
    # Load config and enable Flash Attention 2
    config = WhisperConfig.from_pretrained(BASE_MODEL)
    config.use_flash_attention_2 = True
    
    # Load model with modified config
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        config=config,
        # device_map="auto",  # This helps with device placement
        # torch_dtype="auto", torch.float16  # This will use BF16 if available # both generates nan gradients causing training to fail
    )
    print(f'model: {model}')
    
    if FREEZE_FEATURE_EXTRACTOR:
        # Freeze the feature extractor
        print("[INFO] Freezing feature extractor parameters...")
        
        for name, param in model.named_parameters():
            if any(feature_name in name for feature_name in freeze_feature_extractor_layer_names):
                param.requires_grad = False
            
        print("[INFO] Feature extractor has been frozen. Only decoder and other layers will be trained.")
        
    print_trainable_parameters(model)
    
    # model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    
    # Force Moroccan Arabic (Darija) output
    if FORCE_ARABIC_OUTPUTS:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids
        model.config.suppress_tokens = []
        
        # Ensure the model always uses Moroccan Arabic language token
        model.config.forced_language_ids = processor.tokenizer.get_vocab()["<|ar|>"]
        
        # Define generation config for consistent Moroccan Arabic output
        model.generation_config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.suppress_tokens = []
        model.generation_config.forced_language_ids = processor.tokenizer.get_vocab()["<|ar|>"]
    else:
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

    print(f"[INFO] Loading dataset...")
    training_dataset = load_dataset(TRAIN_DATA_PATH, split="train")
    validation_dataset = load_dataset(TEST_DATA_PATH, split="validation")
    
    if DEBUG:
        # Select only batch_size samples from train and validation set to minimize load and train time, just to see if it runs smoothly
        print(f"Reducing train set to {batch_size} samples...")
        training_dataset = training_dataset.select(range(batch_size))
        
        print(f"Reducing validation set to {batch_size} samples...")
        validation_dataset = validation_dataset.select(range(batch_size))
        
        
    dataset = DatasetDict({
        "train": training_dataset,
        "validation": validation_dataset,
    })
    
    print("[INFO] Preparing datasets...")
    processed_dataset = prepare_datasets(dataset, processor)


    print("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        # tokenizer=processor.feature_extractor,
        processing_class=processor,
        compute_metrics=lambda pred: compute_metrics(pred, processor)
    )

    print("Starting training...")
    trainer.train()

    print("Saving model and processor...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("Pushing to hub...")
    trainer.push_to_hub(PUSH_FINETUNED_MODEL_TO)