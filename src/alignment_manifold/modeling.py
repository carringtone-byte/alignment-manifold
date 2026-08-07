from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def hf_token() -> str | None:
    """Return the process-scoped token without logging or serializing it."""
    return os.environ.get("HF_TOKEN")


def resolve_checkpoint(repo_id: str, revision: str) -> dict[str, str]:
    info = HfApi(token=hf_token()).model_info(repo_id, revision=revision)
    return {"repo_id": repo_id, "requested_revision": revision, "resolved_commit": info.sha}


def load_tokenizer(repo_id: str, revision: str, cache_dir: str | Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        revision=revision,
        cache_dir=str(cache_dir),
        token=hf_token(),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"Checkpoint tokenizer has no chat template: {repo_id}@{revision}")
    return tokenizer


def load_model(
    repo_id: str,
    revision: str,
    cache_dir: str | Path,
    dtype_name: str,
    device: str,
    quantization: str | None = None,
    double_quant: bool = True,
    use_safetensors: bool | None = None,
    device_map: Any | None = None,
    max_memory: dict[Any, str] | None = None,
) -> Any:
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype_name}; choose from {sorted(DTYPES)}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dtype = DTYPES[dtype_name]
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "cache_dir": str(cache_dir),
        "token": hf_token(),
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if use_safetensors is not None:
        model_kwargs["use_safetensors"] = use_safetensors
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    if quantization is None:
        model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
        if device_map is None:
            model.to(device)
    elif quantization == "nf4":
        if device != "cuda":
            raise ValueError("NF4 extraction requires a CUDA device")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    model.eval()
    model.config.use_cache = False
    return model


def release_model(model: Any | None) -> None:
    if model is not None:
        if not getattr(model, "is_loaded_in_4bit", False) and not getattr(
            model, "hf_device_map", None
        ):
            model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transformer_layers(model: Any) -> Any:
    candidates = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("transformer", "blocks"),
    ]
    for candidate in candidates:
        value = model
        try:
            for name in candidate:
                value = getattr(value, name)
        except AttributeError:
            continue
        if value is not None:
            return value
    raise AttributeError(f"Could not locate transformer layers on {type(model).__name__}")
