import os
import wandb
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)
from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader

@dataclass
class PhaseConfig:
    """Configuration for a training phase"""
    phase_id: str
    num_epochs: int
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    
class PhaseTrackingCallback(TrainerCallback):
    """Callback to track phase information and maintain continuous logging"""
    def __init__(self, phase_id: str, initial_global_step: int, initial_epoch: float):
        self.phase_id = phase_id
        self.initial_global_step = initial_global_step
        self.initial_epoch = initial_epoch
        
    def on_log(self, args, state, control, logs, **kwargs):
        """Add phase information to logs and adjust global step and epoch"""
        if logs is not None:
            logs["phase"] = self.phase_id
            
            # Adjust global step to maintain continuity
            if "global_step" in logs:
                logs["global_step"] += self.initial_global_step
                
            # Adjust epoch to maintain continuity
            if "epoch" in logs:
                logs["epoch"] += self.initial_epoch
                
            wandb.log(logs)

class MultiPhaseTrainer:
    def __init__(self, model, phase_configs, base_training_args, compute_metrics, 
                 data_collator, processor, original_dataset, augment_fn, prepare_dataset_fn):
        self.model = model
        self.phase_configs = phase_configs
        self.base_training_args = base_training_args
        self.compute_metrics = compute_metrics
        self.data_collator = data_collator
        self.processor = processor
        self.original_dataset = original_dataset
        self.augment_fn = augment_fn
        self.prepare_dataset_fn = prepare_dataset_fn
        
        # Calculate total epochs
        self.total_epochs = sum(config.num_epochs for config in phase_configs.values())

    def _get_phase_training_args(self, phase_config: PhaseConfig) -> Seq2SeqTrainingArguments:
        """Create training arguments for a specific phase"""
        # Get base arguments as dictionary
        args_dict = vars(self.base_training_args).copy()
        
        # Remove any non-serializable attributes
        args_dict = {k: v for k, v in args_dict.items() 
                    if not k.startswith('__') and not callable(v)}
        
        # Update with phase-specific values
        updates = {
            "num_train_epochs": phase_config.num_epochs,
            "output_dir": os.path.join(self.base_training_args.output_dir, f"phase_{phase_config.phase_id}"),
        }
        
        # Add optional phase-specific parameters if provided
        if phase_config.learning_rate is not None:
            updates["learning_rate"] = phase_config.learning_rate
        if phase_config.batch_size is not None:
            updates["per_device_train_batch_size"] = phase_config.batch_size
            updates["per_device_eval_batch_size"] = phase_config.batch_size
        if phase_config.gradient_accumulation_steps is not None:
            updates["gradient_accumulation_steps"] = phase_config.gradient_accumulation_steps
            
        args_dict.update(updates)
        
        # Create new training arguments
        return Seq2SeqTrainingArguments(**args_dict)
        
    def train(self):
        """Execute multi-phase training"""
        current_global_step = 0
        current_epoch = 0.0
        
        for phase_id, phase_config in self.phase_configs.items():
            print(f"\n=== Starting Phase {phase_id} ===")
            print(f"Number of epochs: {phase_config.num_epochs}")
            print(f"Starting from epoch: {current_epoch}")
            
            # Prepare phase-specific dataset
            train_dataset = self.augment_fn(self.original_dataset['train'], int(phase_id))
            processed_datasets = self.prepare_dataset_fn(
                {"train": train_dataset, "test": self.original_dataset['test']},
                self.processor
            )
            
            # Get phase-specific training arguments
            training_args = self._get_phase_training_args(phase_config)
            
            # Create trainer for this phase
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[
                    PhaseTrackingCallback(
                        phase_id=phase_id,
                        initial_global_step=current_global_step,
                        initial_epoch=current_epoch
                    )
                ]
            )
            
            # Train for this phase
            train_result = trainer.train()
            
            # Update tracking variables
            current_global_step = train_result.global_step
            current_epoch += phase_config.num_epochs
            
            # Save phase checkpoint
            trainer.save_model(
                os.path.join(self.base_training_args.output_dir, f"checkpoint-phase-{phase_id}")
            )
            
            # Log phase completion metrics
            wandb.log({
                f"phase_{phase_id}_final_loss": train_result.training_loss,
                "current_phase": phase_id,
                "total_epochs_completed": current_epoch
            })
            
            # Clean up to free memory
            del trainer
            torch.cuda.empty_cache()
            
        print("\n=== Multi-phase training completed ===")
        
        
from transformers import Seq2SeqTrainer

from dataset import (
    augment_dataset,
    prepare_datasets,
)

class CurriculumTrainer(Seq2SeqTrainer):
    def __init__(self, phase_epochs_dict, original_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_dataset = original_dataset
        self.phase_epochs = list(phase_epochs_dict.items())
        self.cumulative_epochs = []
        current_total = 0
        for phase, n_epochs in self.phase_epochs:
            current_total += n_epochs
            self.cumulative_epochs.append(current_total)
            print(f'[INFO] cumulative_epochs: {self.cumulative_epochs}')
        self.current_phase = None
        self.current_train_dataset = None

    def get_train_dataloader(self):
        # Determine current phase based on self.state.epoch
        current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0
        phase_idx = 0
        print(f'[INFO] current_epoch: {self}')
        print(f'[INFO] cumulative_epochs: {self.cumulative_epochs}')
        
        for i, cumulative in enumerate(self.cumulative_epochs):
            if current_epoch < cumulative:
                phase_idx = i
                break
        else:
            phase_idx = len(self.phase_epochs) - 1

        phase, n_epochs = self.phase_epochs[phase_idx]

        # Update dataset if phase changed
        if phase != self.current_phase:
            self.current_phase = phase
            print(f"[INFO] Switching to phase {phase} at epoch {current_epoch}")
            self.train_dataset = augment_dataset(self.original_dataset['train'], int(phase))
            processed_dataset = prepare_datasets({
                "train": self.train_dataset,
                "test": self.original_dataset['test'],
            }, self.data_collator.processor)
            self.train_dataset = processed_dataset["train"]
            self.eval_dataset = processed_dataset["test"]
            # Reset dataloader to apply new dataset
            self._train_dataloader = None

        return super().get_train_dataloader()