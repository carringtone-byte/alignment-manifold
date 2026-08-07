from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alignment_manifold.causal import (  # noqa: E402
    _intervened_logits,
    _random_rotated_vectors,
    normalized_gap_recovery,
    paired_bidirectional_bootstrap,
)
from alignment_manifold.config import load_config  # noqa: E402
from alignment_manifold.extract import (  # noqa: E402
    assert_matched_extractions,
    load_extraction,
)
from alignment_manifold.geometry import (  # noqa: E402
    _fit_autoencoder,
    _fit_local_pca,
    _fit_pca,
    _project,
    _reconstruct_autoencoder,
    _reconstruct_local,
    _split_indices,
    reconstruction_metrics,
)
from alignment_manifold.modeling import (  # noqa: E402
    load_model,
    load_tokenizer,
    release_model,
)
from alignment_manifold.prompts import load_jsonl  # noqa: E402
from alignment_manifold.provenance import runtime_manifest, write_json  # noqa: E402


def clear_released_model() -> None:
    """Collect caller-owned model references before loading another 4-bit model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_manifest_commit(config: dict[str, Any], checkpoint: str) -> str:
    path = ROOT / config["experiment"]["artifact_dir"] / "extractions" / f"{checkpoint}.manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["checkpoint"]["resolved_commit"]


def load_checkpoint(
    config: dict[str, Any], checkpoint: str, cache_dir: Path
) -> tuple[Any, Any]:
    spec = config["checkpoints"][checkpoint]
    commit = load_manifest_commit(config, checkpoint)
    tokenizer = load_tokenizer(spec["repo_id"], commit, cache_dir)
    extraction = config["extraction"]
    model = load_model(
        spec["repo_id"],
        commit,
        cache_dir,
        extraction["dtype"],
        extraction["device"],
        extraction.get("quantization"),
        bool(extraction.get("double_quant", True)),
        extraction.get("use_safetensors"),
    )
    return tokenizer, model


def baseline_logits(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    extraction: dict[str, np.ndarray],
    indices: np.ndarray,
    layer: int,
    width: int,
    max_length: int,
    device: str,
    label: str,
) -> np.ndarray:
    zeros = np.zeros((len(records), width), dtype=np.float32)
    return _intervened_logits(
        model,
        tokenizer,
        records,
        extraction,
        indices,
        layer,
        zeros,
        0.0,
        1.0,
        max_length,
        device,
        label,
    )


def fit_methods(
    delta: np.ndarray,
    split: dict[str, np.ndarray],
    layer: int,
    seed: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    values = delta[:, layer]
    train = values[split["train"]]
    validation = values[split["validation"]]
    test = values[split["test"]]
    methods: dict[str, np.ndarray] = {"full_delta": values.astype(np.float32)}
    geometry = {}
    selected_mean = None
    selected_basis = None
    for rank in (0, 1, 2, 4, 8, 16, 32, 64):
        if rank == 0:
            mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
            prediction = np.broadcast_to(mean, values.shape).copy()
            basis = np.zeros((0, values.shape[1]), dtype=np.float32)
        else:
            mean, basis = _fit_pca(train, rank, seed + 5000 + rank)
            prediction = _project(values, mean, basis).astype(np.float32)
        methods[f"pca_rank_{rank}"] = prediction
        geometry[f"pca_rank_{rank}"] = {
            "validation": reconstruction_metrics(
                validation, prediction[split["validation"]], mean
            ),
            "test": reconstruction_metrics(test, prediction[split["test"]], mean),
        }
        if rank == 32:
            selected_mean, selected_basis = mean, basis
    if selected_mean is None or selected_basis is None:
        raise AssertionError("Rank-32 PCA fit missing")

    local_model = _fit_local_pca(train, 32, 4, seed + 6000)
    local_prediction = _reconstruct_local(values, local_model).astype(np.float32)
    methods["local_pca_4"] = local_prediction
    geometry["local_pca_4"] = {
        "validation": reconstruction_metrics(
            validation, local_prediction[split["validation"]], selected_mean
        ),
        "test": reconstruction_metrics(
            test, local_prediction[split["test"]], selected_mean
        ),
    }

    autoencoder, ae_mean, ae_scale, ae_training = _fit_autoencoder(
        train, validation, 32, seed + 7000, device
    )
    ae_prediction = _reconstruct_autoencoder(
        autoencoder, values, ae_mean, ae_scale, device
    ).astype(np.float32)
    autoencoder.to("cpu")
    del autoencoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    methods["autoencoder_32"] = ae_prediction
    geometry["autoencoder_32"] = {
        "training": ae_training,
        "validation": reconstruction_metrics(
            validation, ae_prediction[split["validation"]], selected_mean
        ),
        "test": reconstruction_metrics(test, ae_prediction[split["test"]], selected_mean),
    }

    pca32 = methods["pca_rank_32"]
    for random_index in range(5):
        methods[f"random_rotated_{random_index}"] = _random_rotated_vectors(
            pca32,
            selected_mean,
            selected_basis,
            seed + 8000 + random_index,
        )
    return methods, geometry


def run_interventions(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    extraction: dict[str, np.ndarray],
    indices: np.ndarray,
    methods: dict[str, np.ndarray],
    layer: int,
    strength: float,
    sign: float,
    config: dict[str, Any],
    label: str,
) -> dict[str, np.ndarray]:
    return {
        name: _intervened_logits(
            model,
            tokenizer,
            records,
            extraction,
            indices,
            layer,
            vectors,
            strength,
            sign,
            int(config["data"]["max_length"]),
            config["extraction"]["device"],
            f"{label} {name} strength={strength}",
        )
        for name, vectors in methods.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trajectory_7b.yaml")
    parser.add_argument("--donor", default="rlvr_0180")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--validation-examples", type=int, default=40)
    parser.add_argument("--test-examples", type=int, default=40)
    parser.add_argument("--output-dir", default="artifacts/trajectory_7b/trajectory_causal")
    args = parser.parse_args()
    started = time.time()
    config = load_config(ROOT / args.config)
    reference_name = config["trajectory"]["reference_checkpoint"]
    reference = load_extraction(config, reference_name)
    donor = load_extraction(config, args.donor)
    assert_matched_extractions(reference, donor)
    records = load_jsonl(ROOT / config["data"]["path"])
    split = _split_indices(
        reference["cluster_ids"],
        float(config["data"]["train_fraction"]),
        float(config["data"]["validation_fraction"]),
        int(config["experiment"]["seed"]),
    )
    validation = split["validation"][: args.validation_examples]
    test = split["test"][: args.test_examples]
    delta = donor["activations"].astype(np.float32) - reference["activations"].astype(
        np.float32
    )
    methods, geometry = fit_methods(
        delta,
        split,
        args.layer,
        int(config["experiment"]["seed"]),
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = ROOT / "artifacts" / "trajectory_7b_causal_model_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    strengths = (0.5, 1.0, 1.5)

    donor_cache = cache_root / args.donor
    donor_tokenizer, donor_model = load_checkpoint(config, args.donor, donor_cache)
    try:
        donor_validation = baseline_logits(
            donor_model,
            donor_tokenizer,
            records,
            donor,
            validation,
            args.layer,
            delta.shape[2],
            int(config["data"]["max_length"]),
            config["extraction"]["device"],
            "donor validation baseline",
        )
        donor_test = baseline_logits(
            donor_model,
            donor_tokenizer,
            records,
            donor,
            test,
            args.layer,
            delta.shape[2],
            int(config["data"]["max_length"]),
            config["extraction"]["device"],
            "donor test baseline",
        )
    finally:
        release_model(donor_model)
        del donor_model
        clear_released_model()

    reference_cache = cache_root / reference_name
    reference_tokenizer, reference_model = load_checkpoint(
        config, reference_name, reference_cache
    )
    try:
        reference_validation = baseline_logits(
            reference_model,
            reference_tokenizer,
            records,
            reference,
            validation,
            args.layer,
            delta.shape[2],
            int(config["data"]["max_length"]),
            config["extraction"]["device"],
            "reference validation baseline",
        )
        reference_test = baseline_logits(
            reference_model,
            reference_tokenizer,
            records,
            reference,
            test,
            args.layer,
            delta.shape[2],
            int(config["data"]["max_length"]),
            config["extraction"]["device"],
            "reference test baseline",
        )
        addition_sweep = []
        for strength in strengths:
            logits = _intervened_logits(
                reference_model,
                reference_tokenizer,
                records,
                reference,
                validation,
                args.layer,
                methods["pca_rank_32"],
                strength,
                1.0,
                int(config["data"]["max_length"]),
                config["extraction"]["device"],
                f"addition validation strength={strength}",
            )
            addition_sweep.append(
                {
                    "strength": strength,
                    "metrics": normalized_gap_recovery(
                        donor_validation, reference_validation, logits
                    ),
                }
            )
        addition_strength = max(
            addition_sweep, key=lambda row: row["metrics"]["aggregate_recovery"]
        )["strength"]
        addition_logits = run_interventions(
            reference_model,
            reference_tokenizer,
            records,
            reference,
            test,
            methods,
            args.layer,
            addition_strength,
            1.0,
            config,
            "addition test",
        )
    finally:
        release_model(reference_model)
        del reference_model
        clear_released_model()

    donor_tokenizer, donor_model = load_checkpoint(config, args.donor, donor_cache)
    try:
        removal_sweep = []
        for strength in strengths:
            logits = _intervened_logits(
                donor_model,
                donor_tokenizer,
                records,
                donor,
                validation,
                args.layer,
                methods["pca_rank_32"],
                strength,
                -1.0,
                int(config["data"]["max_length"]),
                config["extraction"]["device"],
                f"removal validation strength={strength}",
            )
            removal_sweep.append(
                {
                    "strength": strength,
                    "metrics": normalized_gap_recovery(
                        reference_validation, donor_validation, logits
                    ),
                }
            )
        removal_strength = max(
            removal_sweep, key=lambda row: row["metrics"]["aggregate_recovery"]
        )["strength"]
        removal_logits = run_interventions(
            donor_model,
            donor_tokenizer,
            records,
            donor,
            test,
            methods,
            args.layer,
            removal_strength,
            -1.0,
            config,
            "removal test",
        )
    finally:
        release_model(donor_model)
        del donor_model
        clear_released_model()

    addition = {
        name: normalized_gap_recovery(donor_test, reference_test, logits)
        for name, logits in addition_logits.items()
    }
    removal = {
        name: normalized_gap_recovery(reference_test, donor_test, logits)
        for name, logits in removal_logits.items()
    }
    bidirectional = {}
    for name in methods:
        bidirectional[name] = {
            "mean_aggregate_recovery": 0.5
            * (addition[name]["aggregate_recovery"] + removal[name]["aggregate_recovery"]),
            "paired_prompt_bootstrap": paired_bidirectional_bootstrap(
                addition[name],
                removal[name],
                int(config["experiment"]["seed"]) + 90_000,
            ),
        }
    random_recoveries = [
        bidirectional[name]["mean_aggregate_recovery"]
        for name in bidirectional
        if name.startswith("random_rotated_")
    ]
    report = {
        "kind": "trajectory_bidirectional_causal_test",
        "status": "exploratory_followup",
        "reference": reference_name,
        "donor": args.donor,
        "layer": args.layer,
        "validation_examples": int(len(validation)),
        "test_examples": int(len(test)),
        "addition_strength_sweep": addition_sweep,
        "removal_strength_sweep": removal_sweep,
        "selected_addition_strength": addition_strength,
        "selected_removal_strength": removal_strength,
        "geometry": geometry,
        "addition": addition,
        "removal": removal,
        "bidirectional": bidirectional,
        "decision_summary": {
            "full_delta_recovery": bidirectional["full_delta"]["mean_aggregate_recovery"],
            "pca_rank_32_recovery": bidirectional["pca_rank_32"][
                "mean_aggregate_recovery"
            ],
            "local_pca_recovery": bidirectional["local_pca_4"][
                "mean_aggregate_recovery"
            ],
            "autoencoder_recovery": bidirectional["autoencoder_32"][
                "mean_aggregate_recovery"
            ],
            "random_control_mean": float(np.mean(random_recoveries)),
            "random_control_maximum": float(np.max(random_recoveries)),
        },
        "elapsed_seconds": time.time() - started,
        "runtime": runtime_manifest(),
    }
    output = output_dir / "report.json"
    write_json(report, output)
    print(json.dumps(report["decision_summary"], indent=2))
    print(f"Wrote trajectory causal report: {output}")
    for cache_dir in (reference_cache, donor_cache):
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Removed verified causal model cache: {cache_dir}")


if __name__ == "__main__":
    main()
