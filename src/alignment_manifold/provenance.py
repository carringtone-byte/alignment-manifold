from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy
import sklearn
import torch
import transformers

from alignment_manifold.config import stable_json_hash


def runtime_manifest() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "git": git_manifest(),
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        manifest["gpu"] = {
            "name": properties.name,
            "vram_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return manifest


def git_manifest() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    payload = {
        "class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens_map": tokenizer.special_tokens_map,
        "chat_template": getattr(tokenizer, "chat_template", None),
        # The serialized fast-tokenizer backend captures vocabulary, normalizer,
        # pre-tokenizer, post-processor, decoder, and added tokens. Deliberately
        # exclude name_or_path, cache paths, and commit metadata: those made
        # semantically identical SFT/DPO tokenizers hash differently.
        "backend": backend.to_str() if backend is not None else None,
    }
    return stable_json_hash(json.loads(json.dumps(payload, default=str)))


def write_json(value: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
