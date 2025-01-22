from utils import (
    print_trainable_parameters,
    PhaseTrackingCallback,
)
from evaluate import compute_metrics
from dataset import (
    DataCollatorSpeechSeq2SeqWithPadding,
    prepare_datasets,
    DataCollatorSpeechSeq2SeqWithPaddingCurriculum,
    apply_augmentation,
)
import wandb
import os
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
)
from datasets import (
    load_dataset,
    # Audio,
    DatasetDict,
    # interleave_datasets,
)
from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    # Trainer,
)
# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------- #
# ----------------------      Main code      ---------------------- #
# ----------------------------------------------------------------- #

if __name__ == "__main__":
    
    model_size = "large" # base large large-turbo small base tiny
    # current version of training: a version is usually a new training strategy or new data
    version = "v1.6" 
    # in v1.5 we train with frozen F.E in
    # in v1.6, we train all layers and we eval on Casablanca directly and include lr values from Jong Wook Kim
    # index of training model under version
    # 1 we set model.config.forced_decoder_ids = None, model.config.suppress_tokens = [] 
    # 2 we don't and bS= 32
    # 3: BS=16, 
    # 4: BS of 256 and data aug and curr training
    # 5: recover large-v1.4
    # 6: apply curriculum training
    # 7: increase batch size from 64 to 256 for large model
    # 8: increase batch size from 256 to 512 for large model
    # 9: use additive augmented samples in training set so more data is used from the augmented samples
    
    index_model = "9" 
    # project name that will appear in WandB
    project_name = f'{model_size}_{version}'
    # metric that indicates best model
    metric_for_best_model = "wer"
    
    # do curriculum training
    # CURRICULUM_TRAINING = True
    CURRICULUM_TRAINING = False
    
    # probability of appliying light, moderate or heavy augmentations per phase
    probabilities_per_phase_dict = {
        "0": [0.7, 0.2, 0.1],  # Light augmentations
        "1": [0.5, 0.4, 0.1],  # Moderate augmentations
        "2": [0.3, 0.35, 0.35]  # Heavier augmentations
    }

    # add new augmented samples to the training set
    DO_ADD_AUGMENTED_SAMPLES = True
    # DO_ADD_AUGMENTED_SAMPLES = False
    num_augmented_per_sample= 4
    phase_probabilities= [0.45, 0.35, 0.2]
    
    
    # DEBUG = True
    DEBUG = False
    
    # HF final hub
    PUSH_FINETUNED_MODEL_TO = f"BounharAbdelaziz/Moroccan-Darija-STT-{model_size}-{version}.{index_model}"
    
    # if we freeze F.E
    FREEZE_FEATURE_EXTRACTOR = False

    # layers to freeze during finetuning
    freeze_feature_extractor_layer_names = [
        'model.encoder.conv1.',
        'model.encoder.conv2.',
        'model.encoder.embed_positions.',
        # 'model.encoder.layers.',
        # 'model.encoder.layer_norm.',
    ]
    
    # Data paths
    TRAIN_DATA_PATH = "BounharAbdelaziz/Youtube-Commons-Moroccan-Darija-14h47"
    TEST_DATA_PATH = "UBC-NLP/Casablanca"
    target_language = "Morocco"
    
    # Models  
    BASE_MODELS = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "large-turbo": "openai/whisper-large-v3-turbo",
        "large": "openai/whisper-large-v3",
    }
    
    # Batch size taken as in v1.4 which gave good results
    BATCH_SIZES = {
        "tiny": 256,
        "base": 256,
        "small": 128,
        "large-turbo": 32,
        "large": 16,
    }
    
    # lr values taken as suggested by Jong Wook Kim, one of the authors of the Whisper paper. See table in https://github.com/vasistalodagala/whisper-finetune
    LEARNING_RATES = {
        "tiny": 3.75e-5,
        "base": 2.5e-5,
        "small": 1.25e-5,
        "large-turbo": 4.375e-6,
        "large": 1e-5,
    }
    
    # number of training epochs
    N_EPOCHS_DICT = {
        "tiny": 2,
        "base": 2,
        "small": 6,
        "large-turbo": 6,
        "large": 6,
    }
    
    # simulate large batches
    GRAD_ACC = {
        "tiny": 1,                  # 256 effective batch size
        "base": 1,                  # 256 effective batch size
        "small": 1,                 # 128 effective batch size
        "large-turbo": 4,           # 128 effective batch size
        "large": 16,                # 256 effective batch size
    }
    
    # Add gradient clipping settings
    MAX_GRAD_NORM = {
        "tiny": 1.0,
        "base": 1.0,
        "small": 1.0,
        "large": 1.0,  # Lower for larger models
        "large-turbo": 1.0,  # Lower for larger models
    }
    
    # # Add gradient clipping settings ## use original value
    # WEIGHT_DECAYS = {
    #     "tiny": 0.0,
    #     "base": 0.0,
    #     "small": 0.001,
    #     "large": 0.005,  
    #     "large-turbo": 0.005, 
    # }
    
    # get base model name
    BASE_MODEL = BASE_MODELS[model_size]
    
    # Saving directory
    OUTPUT_DIR = f"./Mixed/{BASE_MODEL}/Morocco-Darija-STT-{model_size}-{version}.{index_model}"
    OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Apply chmod -R 755 to OUTPUT_DIR
    print(f"[INFO] Applying chmod -R 755 to OUTPUT_DIR: {OUTPUT_DIR}...")
    os.system(f"chmod -R 755 {OUTPUT_DIR}")
    
    # hyperparameters
    total_training_epochs = N_EPOCHS_DICT[model_size]
    
    lr = LEARNING_RATES[model_size]
    batch_size = BATCH_SIZES[model_size]
    # weight_decay = WEIGHT_DECAYS[model_size]
    gradient_accumulation_steps = GRAD_ACC[model_size]
    max_grad_norm = MAX_GRAD_NORM[model_size]  # Get max gradient norm for current model size
    eval_strategy='steps'
    save_strategy='steps'
    # warmup_ratio = 0.1 # 10% of total steps
    warmup_steps = 10
    logging_steps = 5
    eval_steps = 10
    save_steps = 10
    lr_scheduler_type = "linear" #"cosine"
    run_name = f"{model_size}-{version}-ep-{total_training_epochs}-bs-{batch_size}-g_acc-{gradient_accumulation_steps}-additive-data-aug" #-wd-{weight_decay}"

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
            "probabilities_per_phase_dict": probabilities_per_phase_dict,
            "total_training_epochs": total_training_epochs,
            "batch_size": batch_size,
            # "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            # "weight_decay": weight_decay,
            "dataset": TRAIN_DATA_PATH,
        }
    )

    print(f"[INFO] Finetuning base model: {BASE_MODEL}...")
    print(f"[INFO] Using training dataset: {TRAIN_DATA_PATH}...")
    print(f"[INFO] Using test dataset: {TEST_DATA_PATH}...")
    print(f"[INFO] Finetuned model will be pushed to hub: {PUSH_FINETUNED_MODEL_TO}...")
    
    print("[INFO] Loading processor and model...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, task="transcribe")
    
    # Load config and enable Flash Attention 2
    config = WhisperConfig.from_pretrained(BASE_MODEL)
    config.use_flash_attention_2 = True
    # has been shown to improve perfs: v1.6.1 confirms this -> we keep original behavior in v1.6.6
    # config.return_timestamps  = False
    
    # Load model with modified config
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        config=config,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = [] # don't use these to be able to set the language later?

    print(f'[INFO] model: {model}')
    
    if FREEZE_FEATURE_EXTRACTOR:
        # Freeze the feature extractor
        print("[INFO] Freezing feature extractor layers...")
        
        for name, param in model.named_parameters():
            if any(feature_name in name for feature_name in freeze_feature_extractor_layer_names):
                param.requires_grad = False
            
        print("[INFO] Feature extractor has been frozen. Only decoder and other layers will be trained.")
        
    # show number of trainable parameters
    print_trainable_parameters(model)

    # Load train and test data
    print(f"[INFO] Loading dataset...")
    training_dataset = load_dataset(TRAIN_DATA_PATH, split="train")
    test_dataset = load_dataset(TEST_DATA_PATH, target_language, split="test")
    
    if DEBUG:
        # Select only batch_size samples from train and validation set to minimize load and train time, just to see if it runs smoothly
        print(f"Reducing train set to {batch_size} samples...")
        training_dataset = training_dataset.select(range(batch_size))
        
        print(f"Reducing test set to {batch_size} samples...")
        test_dataset = test_dataset.select(range(batch_size))
    
    # wrap train and test data in a single dataset
    dataset = DatasetDict({
        "train": training_dataset,
        "test": test_dataset,
    })
    
    # Training parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        learning_rate=lr,
        # optim=optimizer, # default is adamw
        lr_scheduler_type=lr_scheduler_type,  # Linear learning rate scheduler
        # warmup_ratio=warmup_ratio,  # Warmup for the first steps
        warmup_steps=warmup_steps, # Warmup for the first steps
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # weight_decay=weight_decay,
        save_total_limit=1,
        num_train_epochs=total_training_epochs,
        predict_with_generate=True,
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
        push_to_hub=False,  
        hub_model_id=PUSH_FINETUNED_MODEL_TO,
        hub_strategy="every_save",  # Push model to hub after every save
        remove_unused_columns=False,
    )

    print("[INFO] Initializing trainer...")
    if CURRICULUM_TRAINING:
        print("[INFO] Curriculum training enabled with dataset interleaving...")
        
        # Calculate total training steps
        num_train_examples = len(dataset["train"])

        # Calculate total steps
        batch_size = training_args.per_device_train_batch_size
        gradient_accumulation_steps = training_args.gradient_accumulation_steps or 1
        num_devices = 1  # Only 1 GPU, Adjust if using multiple GPUs

        steps_per_epoch = (num_train_examples // (batch_size * gradient_accumulation_steps * num_devices)) + 1
        total_steps = steps_per_epoch * training_args.num_train_epochs

        print(f"[INFO] Total training steps: {total_steps}")
        
        # Initialize the data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPaddingCurriculum(
            processor=processor,
            total_steps=total_steps,
            probabilities_per_phase_dict=probabilities_per_phase_dict,
        )
        
        # Initialize the callback
        phase_tracking_callback = PhaseTrackingCallback(total_steps=total_steps)

        # Initialize the Trainer with the callback
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_metrics(pred, processor),
            callbacks=[phase_tracking_callback]  # Add the callback
        )
        
    else:
        
        if DO_ADD_AUGMENTED_SAMPLES:
            print("[INFO] Curriculum training disabled, but adding augmented samples to training set...")
            print(f"[INFO] Started augmenting training dataset, original size {len(dataset['train'])}...")
            
            # apply data augmentation
            dataset["train"] = apply_augmentation(dataset["train"], num_augmented_per_sample, phase_probabilities)
            print(f"[INFO] Ended augmenting training dataset, new size {len(dataset['train'])}...")
            
        else:
            print("[INFO] Curriculum training disabled...")
            
        print("[INFO] Preparing datasets...")            
        # prepare data for training
        processed_dataset = prepare_datasets(dataset, processor)
        
        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["test"],
            data_collator=data_collator,
            processing_class=processor,
            compute_metrics=lambda predictions: compute_metrics(predictions, processor),
        )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving model and processor...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("[INFO] Pushing to hub...")
    trainer.push_to_hub(PUSH_FINETUNED_MODEL_TO)