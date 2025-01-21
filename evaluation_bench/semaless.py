from transformers import Pipeline, AutoProcessor, SeamlessM4Tv2Model
import torch
import torchaudio
from typing import Union, Dict, Any
import numpy as np
from pathlib import Path

class SeamlessM4TPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_rate = 16000

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}

        # Handle source and target languages
        forward_params["tgt_lang"] = kwargs.pop("tgt_lang", "rus")
        preprocess_params["src_lang"] = kwargs.pop("src_lang", "eng")

        # Return format
        postprocess_params["return_tensors"] = kwargs.pop("return_tensors", False)

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs: Union[str, Path, torch.Tensor, np.ndarray], src_lang: str = "eng") -> Dict:
        if isinstance(inputs, str):
            # Check if it's a file path or text
            if Path(inputs).exists():
                audio, orig_freq = torchaudio.load(inputs)
                if orig_freq != self.sample_rate:
                    audio = torchaudio.functional.resample(
                        audio, 
                        orig_freq=orig_freq, 
                        new_freq=self.sample_rate
                    )
                return self.tokenizer(audios=audio, return_tensors="pt")
            else:
                return self.tokenizer(text=inputs, src_lang=src_lang, return_tensors="pt")
        elif isinstance(inputs, (torch.Tensor, np.ndarray)):
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs)
            return self.tokenizer(audios=inputs, return_tensors="pt")
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

    def _forward(self, model_inputs: Dict, tgt_lang: str = "rus") -> torch.Tensor:
        return self.model.generate(**model_inputs, tgt_lang=tgt_lang)[0]

    def postprocess(self, model_outputs: torch.Tensor, return_tensors: bool = False) -> Union[torch.Tensor, np.ndarray]:
        if return_tensors:
            return model_outputs
        return model_outputs.cpu().numpy().squeeze()

def register_seamless_pipeline():
    """Register the Seamless M4T pipeline with the transformers library."""
    from transformers import pipeline
    Pipeline.register_for_auto_class("seamless-m4t")
    return lambda model: SeamlessM4TPipeline(
        model=model,
        tokenizer=AutoProcessor.from_pretrained(model.config._name_or_path),
        feature_extractor=None,
        framework="pt"
    )