from transformers import TrainerCallback

class PhaseTrackingCallback(TrainerCallback):
    """Callback to track phase information and maintain continuous logging"""
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0

    def get_phase(self):
        """Determine the current phase based on progress."""
        progress = self.current_step / self.total_steps
        
        if progress < 0.33:  # First third of training
            return "light"
        elif progress < 0.66:  # Middle third of training
            return "moderate"
        else:  # Final third of training
            return "heavy"

    def on_step_end(self, args, state, control, **kwargs):
        """Update the current step and phase after each step."""
        self.current_step = state.global_step
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add phase and progress information to logs."""
        if logs is not None:
            # Calculate progress
            progress = self.current_step / self.total_steps
            logs["progress"] = progress
            
            # Determine the current phase
            phase = self.get_phase()
            logs["augmentation_phase"] = phase
            
            # Log to WandB (if enabled)
            if args.report_to == "wandb":
                import wandb
                wandb.log(logs)
        
        return control

def print_trainable_parameters(model):
    """ Prints the number of trainable parameters in the model and their names """
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f'[INFO] Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of all parameters)')