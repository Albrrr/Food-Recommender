"""
Initializes the LLM.

Defaults to Llama 3B Instruct (configurable via env var MODEL_NAME).
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# User request: Llama 3B Instruct. This is the commonly used HF model id.
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def load_model():
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Export your Hugging Face token (must have access to the Meta Llama model)."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    use_cuda = torch.cuda.is_available()
    model_kwargs = {
        "token": hf_token,
        # device_map="auto" needs accelerate; on CPU it can be slow / problematic
        "device_map": "auto" if use_cuda else None,
        "torch_dtype": torch.float16 if use_cuda else None,
    }
    # Drop None values (transformers doesn't like explicit None for some args)
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer
