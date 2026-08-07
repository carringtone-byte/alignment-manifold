from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alignment_manifold.config import artifact_dir, sha256_file, stable_json_hash
from alignment_manifold.modeling import (
    load_model,
    load_tokenizer,
    release_model,
    resolve_checkpoint,
)
from alignment_manifold.prompts import load_jsonl
from alignment_manifold.provenance import runtime_manifest, tokenizer_fingerprint, write_json


def _serialize_prompt(tokenizer: Any, prompt: str, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    if encoded["input_ids"].shape[0] != 1:
        raise AssertionError("Smoke extraction expects one serialized prompt at a time")
    return encoded


def _token_hash(token_ids: np.ndarray) -> str:
    return hashlib.sha256(token_ids.astype(np.int32, copy=False).tobytes()).hexdigest()


def extraction_paths(config: dict[str, Any], checkpoint_name: str) -> tuple[Path, Path]:
    root = artifact_dir(config) / "extractions"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{checkpoint_name}.npz", root / f"{checkpoint_name}.manifest.json"


def extract_checkpoint(
    config: dict[str, Any], checkpoint_name: str, force: bool = False
) -> dict[str, Any]:
    if checkpoint_name not in config["checkpoints"]:
        raise KeyError(f"Unknown checkpoint {checkpoint_name}")
    output_path, manifest_path = extraction_paths(config, checkpoint_name)
    if output_path.exists() and manifest_path.exists() and not force:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        print(f"Using existing extraction: {output_path}")
        return manifest

    checkpoint = config["checkpoints"][checkpoint_name]
    data_config = config["data"]
    extraction_config = config["extraction"]
    records = load_jsonl(data_config["path"])
    expected = int(data_config["expected_examples"])
    if len(records) != expected:
        raise ValueError(f"Expected {expected} prompts, found {len(records)}")

    ephemeral_cache = bool(checkpoint.get("ephemeral_cache", False))
    if ephemeral_cache:
        ephemeral_root = Path(config["experiment"]["ephemeral_cache_root"]).resolve()
        cache_dir = (ephemeral_root / checkpoint_name).resolve()
        try:
            cache_dir.relative_to(ephemeral_root)
        except ValueError as error:
            raise ValueError(f"Ephemeral cache escaped configured root: {cache_dir}") from error
        if cache_dir == ephemeral_root:
            raise ValueError("An ephemeral checkpoint cache cannot equal its cache root")
    else:
        cache_dir = Path(config["experiment"]["hf_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_checkpoint(checkpoint["repo_id"], checkpoint["revision"])
    tokenizer = load_tokenizer(checkpoint["repo_id"], resolved["resolved_commit"], cache_dir)
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    model = None
    started = time.time()
    try:
        model = load_model(
            checkpoint["repo_id"],
            resolved["resolved_commit"],
            cache_dir,
            extraction_config["dtype"],
            extraction_config["device"],
            extraction_config.get("quantization"),
            bool(extraction_config.get("double_quant", True)),
            extraction_config.get("use_safetensors"),
            extraction_config.get("device_map"),
            extraction_config.get("max_memory"),
        )
        layer_count = int(model.config.num_hidden_layers)
        hidden_size = int(model.config.hidden_size)
        vocab_size = int(model.config.vocab_size)
        activations = np.empty((len(records), layer_count, hidden_size), dtype=np.float16)
        position_fractions = tuple(
            float(value) for value in extraction_config.get("position_fractions", [])
        )
        position_activations = (
            np.empty(
                (len(records), len(position_fractions), layer_count, hidden_size),
                dtype=np.float16,
            )
            if position_fractions
            else None
        )
        position_indices = (
            np.empty((len(records), len(position_fractions)), dtype=np.int32)
            if position_fractions
            else None
        )
        store_generated = bool(extraction_config.get("store_first_generated_activation", False))
        generated_activations = (
            np.empty((len(records), layer_count, hidden_size), dtype=np.float16)
            if store_generated
            else None
        )
        generated_token_ids = (
            np.empty(len(records), dtype=np.int32) if store_generated else None
        )
        logits = (
            np.empty((len(records), vocab_size), dtype=np.float16)
            if extraction_config.get("store_logits", True)
            else None
        )
        token_hashes: list[str] = []
        token_lengths = np.empty(len(records), dtype=np.int32)
        flat_token_ids: list[np.ndarray] = []
        offsets = [0]
        device = extraction_config["device"]

        with torch.inference_mode():
            for index, record in enumerate(records):
                encoded_cpu = _serialize_prompt(tokenizer, record["prompt"], data_config["max_length"])
                ids_np = encoded_cpu["input_ids"][0].cpu().numpy().astype(np.int32, copy=False)
                token_hashes.append(_token_hash(ids_np))
                token_lengths[index] = len(ids_np)
                flat_token_ids.append(ids_np)
                offsets.append(offsets[-1] + len(ids_np))
                encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
                output = model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states = output.hidden_states
                if len(hidden_states) != layer_count + 1:
                    raise AssertionError(
                        f"Expected {layer_count + 1} hidden-state tensors, got {len(hidden_states)}"
                    )
                last_position = int(encoded["attention_mask"][0].sum().item() - 1)
                stacked = torch.stack(
                    [state[0, last_position].float().cpu() for state in hidden_states[1:]],
                    dim=0,
                )
                activations[index] = stacked.numpy().astype(np.float16)
                if position_activations is not None and position_indices is not None:
                    sequence_length = last_position + 1
                    for fraction_index, fraction in enumerate(position_fractions):
                        if not 0.0 < fraction <= 1.0:
                            raise ValueError(
                                f"position_fractions must be in (0, 1], got {fraction}"
                            )
                        position = min(
                            last_position,
                            max(0, int(round((sequence_length - 1) * fraction))),
                        )
                        position_indices[index, fraction_index] = position
                        position_activations[index, fraction_index] = torch.stack(
                            [state[0, position].float().cpu() for state in hidden_states[1:]],
                            dim=0,
                        ).numpy().astype(np.float16)
                if logits is not None:
                    logits[index] = (
                        output.logits[0, last_position].float().cpu().numpy().astype(np.float16)
                    )
                if generated_activations is not None and generated_token_ids is not None:
                    next_token = output.logits[0, last_position].argmax().reshape(1, 1)
                    generated_token_ids[index] = int(next_token.item())
                    generated = {
                        "input_ids": torch.cat((encoded["input_ids"], next_token), dim=1),
                        "attention_mask": torch.cat(
                            (
                                encoded["attention_mask"],
                                torch.ones(
                                    (1, 1),
                                    dtype=encoded["attention_mask"].dtype,
                                    device=encoded["attention_mask"].device,
                                ),
                            ),
                            dim=1,
                        ),
                    }
                    generated_output = model(
                        **generated,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    generated_activations[index] = torch.stack(
                        [state[0, -1].float().cpu() for state in generated_output.hidden_states[1:]],
                        dim=0,
                    ).numpy().astype(np.float16)
                    del generated_output, generated, next_token
                del output, hidden_states, stacked, encoded
                if (index + 1) % 10 == 0 or index + 1 == len(records):
                    print(f"[{checkpoint_name}] extracted {index + 1}/{len(records)}")

        arrays: dict[str, Any] = {
            "activations": activations,
            "example_ids": np.asarray([r["example_id"] for r in records], dtype="U20"),
            "cluster_ids": np.asarray([r["cluster_id"] for r in records], dtype="U32"),
            "categories": np.asarray([r["category"] for r in records], dtype="U32"),
            "token_hashes": np.asarray(token_hashes, dtype="U64"),
            "token_lengths": token_lengths,
            "token_ids": np.concatenate(flat_token_ids).astype(np.int32, copy=False),
            "token_offsets": np.asarray(offsets, dtype=np.int64),
        }
        if logits is not None:
            arrays["logits"] = logits
        if position_activations is not None and position_indices is not None:
            arrays["position_activations"] = position_activations
            arrays["position_fractions"] = np.asarray(position_fractions, dtype=np.float32)
            arrays["position_indices"] = position_indices
        if generated_activations is not None and generated_token_ids is not None:
            arrays["generated_activations"] = generated_activations
            arrays["generated_token_ids"] = generated_token_ids
        np.savez(output_path, **arrays)
    finally:
        release_model(model)
        model = None

    manifest = {
        "kind": "matched_activation_extraction",
        "checkpoint_name": checkpoint_name,
        "checkpoint": resolved,
        "checkpoint_role": checkpoint["role"],
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "data_path": str(Path(data_config["path"]).resolve()),
        "data_sha256": sha256_file(data_config["path"]),
        "record_order_hash": stable_json_hash([r["example_id"] for r in records]),
        "tokenizer_hash": tokenizer_hash,
        "examples": len(records),
        "layers": layer_count,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "dtype": extraction_config["dtype"],
        "device": extraction_config["device"],
        "token_regime": extraction_config["token_regime"],
        "position_fractions": list(position_fractions),
        "store_first_generated_activation": store_generated,
        "quantization": extraction_config.get("quantization"),
        "double_quant": bool(extraction_config.get("double_quant", True)),
        "cache_policy": "ephemeral_per_checkpoint" if ephemeral_cache else "persistent",
        "elapsed_seconds": time.time() - started,
        "artifact_path": str(output_path.resolve()),
        "artifact_sha256": sha256_file(output_path),
        "runtime": runtime_manifest(),
    }
    write_json(manifest, manifest_path)
    if ephemeral_cache and cache_dir.exists():
        # The artifact and its checksum are durable before this generated cache
        # is removed. Strict path validation above prevents deleting the shared
        # cache, workspace root, or any out-of-workspace location.
        shutil.rmtree(cache_dir)
        print(f"Removed verified ephemeral cache: {cache_dir}")
    print(f"Wrote extraction: {output_path}")
    return manifest


def load_extraction(config: dict[str, Any], checkpoint_name: str) -> dict[str, np.ndarray]:
    output_path, _ = extraction_paths(config, checkpoint_name)
    if not output_path.exists():
        raise FileNotFoundError(f"Missing extraction for {checkpoint_name}: {output_path}")
    with np.load(output_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def assert_matched_extractions(parent: dict[str, np.ndarray], donor: dict[str, np.ndarray]) -> None:
    for key in ("example_ids", "cluster_ids", "token_hashes", "token_lengths", "token_ids", "token_offsets"):
        if not np.array_equal(parent[key], donor[key]):
            raise AssertionError(f"Checkpoint extractions are not matched for field: {key}")
