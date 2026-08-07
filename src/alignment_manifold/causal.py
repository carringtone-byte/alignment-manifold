from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.special import logsumexp

from alignment_manifold.config import artifact_dir
from alignment_manifold.extract import load_extraction
from alignment_manifold.geometry import (
    _bootstrap_stability,
    _fit_autoencoder,
    _fit_local_pca,
    _fit_pca,
    _project,
    _reconstruct_autoencoder,
    _reconstruct_local,
    reconstruction_metrics,
)
from alignment_manifold.modeling import (
    load_model,
    load_tokenizer,
    release_model,
    transformer_layers,
)
from alignment_manifold.prompts import load_jsonl
from alignment_manifold.provenance import runtime_manifest, write_json


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    return values - logsumexp(values, axis=-1, keepdims=True)


def kl_divergence(target_logits: np.ndarray, candidate_logits: np.ndarray) -> np.ndarray:
    target_log_probs = _log_softmax(target_logits)
    candidate_log_probs = _log_softmax(candidate_logits)
    target_probs = np.exp(target_log_probs)
    return np.sum(target_probs * (target_log_probs - candidate_log_probs), axis=-1)


def normalized_gap_recovery(
    target_logits: np.ndarray,
    baseline_logits: np.ndarray,
    intervention_logits: np.ndarray,
) -> dict[str, float]:
    baseline_kl = kl_divergence(target_logits, baseline_logits)
    intervention_kl = kl_divergence(target_logits, intervention_logits)
    total_baseline = float(np.sum(baseline_kl))
    aggregate = (
        1.0 - float(np.sum(intervention_kl)) / total_baseline
        if total_baseline > 0
        else float("nan")
    )
    per_example = np.divide(
        intervention_kl,
        baseline_kl,
        out=np.full_like(intervention_kl, np.nan),
        where=baseline_kl > 1e-12,
    )
    per_example = 1.0 - per_example
    return {
        "aggregate_recovery": aggregate,
        "mean_recovery": float(np.nanmean(per_example)),
        "median_recovery": float(np.nanmedian(per_example)),
        "mean_baseline_kl": float(np.mean(baseline_kl)),
        "mean_intervention_kl": float(np.mean(intervention_kl)),
        "per_example_baseline_kl": baseline_kl.tolist(),
        "per_example_intervention_kl": intervention_kl.tolist(),
        "per_example_recovery": per_example.tolist(),
    }


def paired_bidirectional_bootstrap(
    addition: dict[str, Any],
    removal: dict[str, Any],
    seed: int,
    samples: int = 2000,
) -> dict[str, float]:
    add_base = np.asarray(addition["per_example_baseline_kl"], dtype=np.float64)
    add_intervention = np.asarray(
        addition["per_example_intervention_kl"], dtype=np.float64
    )
    remove_base = np.asarray(removal["per_example_baseline_kl"], dtype=np.float64)
    remove_intervention = np.asarray(
        removal["per_example_intervention_kl"], dtype=np.float64
    )
    if not (
        len(add_base)
        == len(add_intervention)
        == len(remove_base)
        == len(remove_intervention)
    ):
        raise ValueError("Bidirectional metrics must contain the same number of prompts")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for bootstrap_index in range(samples):
        indices = rng.integers(0, len(add_base), size=len(add_base))
        add_recovery = 1.0 - add_intervention[indices].sum() / add_base[indices].sum()
        remove_recovery = (
            1.0 - remove_intervention[indices].sum() / remove_base[indices].sum()
        )
        estimates[bootstrap_index] = 0.5 * (add_recovery + remove_recovery)
    return {
        "mean": float(np.mean(estimates)),
        "ci_025": float(np.quantile(estimates, 0.025)),
        "ci_975": float(np.quantile(estimates, 0.975)),
        "samples": samples,
    }


def _serialized(tokenizer: Any, prompt: str, max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )


def _assert_tokens_match(encoded: dict[str, torch.Tensor], extraction: dict[str, np.ndarray], index: int) -> None:
    start = int(extraction["token_offsets"][index])
    end = int(extraction["token_offsets"][index + 1])
    expected = extraction["token_ids"][start:end]
    actual = encoded["input_ids"][0].cpu().numpy().astype(np.int32, copy=False)
    if not np.array_equal(actual, expected):
        raise AssertionError(f"Serialized tokens changed for example index {index}")


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def _intervened_logits(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    extraction: dict[str, np.ndarray],
    indices: np.ndarray,
    layer: int,
    vectors: np.ndarray,
    strength: float,
    sign: float,
    max_length: int,
    device: str,
    progress_label: str,
) -> np.ndarray:
    layers = transformer_layers(model)
    if layer < 0 or layer >= len(layers):
        raise IndexError(f"Layer {layer} outside 0..{len(layers) - 1}")
    output_logits = np.empty((len(indices), int(model.config.vocab_size)), dtype=np.float16)
    with torch.inference_mode():
        for output_index, example_index in enumerate(indices):
            encoded_cpu = _serialized(tokenizer, records[int(example_index)]["prompt"], max_length)
            _assert_tokens_match(encoded_cpu, extraction, int(example_index))
            encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
            position = int(encoded["attention_mask"][0].sum().item() - 1)
            vector = torch.from_numpy(vectors[int(example_index)].astype(np.float32)).to(
                device=device, dtype=next(model.parameters()).dtype
            )

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                updated = hidden.clone()
                updated[0, position] = updated[0, position] + sign * strength * vector
                return _replace_hidden(output, updated)

            handle = layers[layer].register_forward_hook(hook)
            try:
                result = model(**encoded, use_cache=False, return_dict=True)
            finally:
                handle.remove()
            output_logits[output_index] = (
                result.logits[0, position].float().cpu().numpy().astype(np.float16)
            )
            del result, encoded, vector
            if (output_index + 1) % 10 == 0 or output_index + 1 == len(indices):
                print(f"{progress_label}: {output_index + 1}/{len(indices)}")
    return output_logits


def _random_rotated_vectors(
    delta: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rank = basis.shape[0]
    random_matrix = rng.standard_normal((delta.shape[1], rank + 1), dtype=np.float32)
    random_basis, _ = np.linalg.qr(random_matrix, mode="reduced")
    mean_norm = float(np.linalg.norm(mean))
    coefficients = (delta - mean) @ basis.T
    rotated = mean_norm * random_basis[:, 0][None, :] + coefficients @ random_basis[:, 1:].T
    target_norm = np.linalg.norm(delta, axis=1)
    rotated_norm = np.linalg.norm(rotated, axis=1)
    scale = np.divide(
        target_norm,
        rotated_norm,
        out=np.ones_like(target_norm),
        where=rotated_norm > 0,
    )
    return (rotated * scale[:, None]).astype(np.float32)


def _load_checkpoint_manifest(config: dict[str, Any], name: str) -> dict[str, Any]:
    path = artifact_dir(config) / "extractions" / f"{name}.manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _screen_direction(
    config: dict[str, Any],
    model_name: str,
    sign: float,
    target_logits: np.ndarray,
    baseline_logits: np.ndarray,
    records: list[dict[str, Any]],
    extraction: dict[str, np.ndarray],
    all_layer_delta: np.ndarray,
    indices: np.ndarray,
) -> list[dict[str, Any]]:
    checkpoint = config["checkpoints"][model_name]
    manifest = _load_checkpoint_manifest(config, model_name)
    commit = manifest["checkpoint"]["resolved_commit"]
    cache_dir = Path(config["experiment"]["hf_cache_dir"])
    tokenizer = load_tokenizer(checkpoint["repo_id"], commit, cache_dir)
    model = None
    results = []
    strength = float(config["causal"]["layer_screen_strength"])
    try:
        model = load_model(
            checkpoint["repo_id"],
            commit,
            cache_dir,
            config["extraction"]["dtype"],
            config["extraction"]["device"],
        )
        for layer in range(all_layer_delta.shape[1]):
            logits = _intervened_logits(
                model,
                tokenizer,
                records,
                extraction,
                indices,
                layer,
                all_layer_delta[:, layer],
                strength,
                sign,
                int(config["data"]["max_length"]),
                config["extraction"]["device"],
                f"{model_name} causal-layer screen layer={layer}",
            )
            results.append(
                {
                    "layer": layer,
                    "strength": strength,
                    "full_delta": normalized_gap_recovery(
                        target_logits[indices], baseline_logits[indices], logits
                    ),
                }
            )
        return results
    finally:
        release_model(model)


def _fit_causal_layer_geometry(
    config: dict[str, Any],
    geometry_report: dict[str, Any],
    all_layer_delta: np.ndarray,
    layer: int,
    split: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seed = int(config["experiment"]["seed"])
    layer_rows = [
        row for row in geometry_report["all_layer_rank_results"] if int(row["layer"]) == layer
    ]
    best_validation = max(row["validation"]["r2_about_train_mean"] for row in layer_rows)
    near_best = [
        row
        for row in layer_rows
        if row["validation"]["r2_about_train_mean"] >= best_validation - 0.01
    ]
    selected_row = sorted(
        near_best,
        key=lambda row: (int(row["rank"]), -row["validation"]["r2_about_train_mean"]),
    )[0]
    rank = int(selected_row["rank"])
    values = all_layer_delta[:, layer]
    train = values[split["train"]]
    validation = values[split["validation"]]
    test = values[split["test"]]
    mean, basis = _fit_pca(train, rank, seed + 60_000 + layer)
    pca_all = _project(values, mean, basis).astype(np.float32)
    rank_reconstructions: dict[str, np.ndarray] = {
        "pca_rank_0": np.broadcast_to(mean, values.shape).copy().astype(np.float32)
    }
    rank_curve_geometry = [
        {
            "rank": 0,
            "validation": reconstruction_metrics(
                validation,
                rank_reconstructions["pca_rank_0"][split["validation"]],
                mean,
            ),
            "test": reconstruction_metrics(
                test, rank_reconstructions["pca_rank_0"][split["test"]], mean
            ),
        }
    ]
    for curve_rank_value in config["geometry"]["ranks"]:
        curve_rank = int(curve_rank_value)
        curve_mean, curve_basis = _fit_pca(
            train, curve_rank, seed + 60_500 + layer * 100 + curve_rank
        )
        curve_prediction = _project(values, curve_mean, curve_basis).astype(np.float32)
        rank_reconstructions[f"pca_rank_{curve_rank}"] = curve_prediction
        rank_curve_geometry.append(
            {
                "rank": curve_rank,
                "validation": reconstruction_metrics(
                    validation, curve_prediction[split["validation"]], curve_mean
                ),
                "test": reconstruction_metrics(
                    test, curve_prediction[split["test"]], curve_mean
                ),
            }
        )

    local_candidates = []
    for components_value in config["geometry"]["local_components"]:
        components = int(components_value)
        model = _fit_local_pca(train, rank, components, seed + 61_000 + components)
        validation_prediction = _reconstruct_local(validation, model)
        local_candidates.append(
            {
                "components": components,
                "model": model,
                "validation": reconstruction_metrics(validation, validation_prediction, mean),
            }
        )
    selected_local = max(
        local_candidates, key=lambda item: item["validation"]["r2_about_train_mean"]
    )
    local_all = _reconstruct_local(values, selected_local["model"]).astype(np.float32)

    ae_device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder, ae_mean, ae_scale, ae_training = _fit_autoencoder(
        train, validation, rank, seed + 62_000 + layer, ae_device
    )
    autoencoder_all = _reconstruct_autoencoder(
        autoencoder, values, ae_mean, ae_scale, ae_device
    ).astype(np.float32)
    autoencoder.to("cpu")
    del autoencoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pca_test = reconstruction_metrics(test, pca_all[split["test"]], mean)
    local_test = reconstruction_metrics(test, local_all[split["test"]], mean)
    autoencoder_test = reconstruction_metrics(
        test, autoencoder_all[split["test"]], mean
    )
    report = {
        "layer": layer,
        "rank": rank,
        "rank_selection_rule": "smallest rank within 0.01 validation R2 of this layer's maximum",
        "global_pca": {
            "validation": reconstruction_metrics(
                validation, pca_all[split["validation"]], mean
            ),
            "test": pca_test,
            "bootstrap_stability": _bootstrap_stability(
                train,
                basis,
                rank,
                int(config["geometry"]["bootstrap_samples"]),
                seed + 63_000 + layer,
            ),
        },
        "rank_curve_geometry": rank_curve_geometry,
        "local_pca": {
            "components": selected_local["components"],
            "validation": selected_local["validation"],
            "test": local_test,
        },
        "autoencoder": {
            "training": ae_training,
            "validation": reconstruction_metrics(
                validation, autoencoder_all[split["validation"]], mean
            ),
            "test": autoencoder_test,
        },
        "compact_candidate": bool(pca_test["r2_about_train_mean"] >= 0.70),
        "best_nonlinear_reconstruction_gain": float(
            max(
                local_test["r2_about_train_mean"],
                autoencoder_test["r2_about_train_mean"],
            )
            - pca_test["r2_about_train_mean"]
        ),
    }
    reconstructions = {
        "delta": values.astype(np.float32),
        "pca": pca_all,
        "local": local_all,
        "autoencoder": autoencoder_all,
        "mean": mean.astype(np.float32),
        "basis": basis.astype(np.float32),
        "validation_indices": split["validation"].astype(np.int64),
        "test_indices": split["test"].astype(np.int64),
    }
    reconstructions.update(rank_reconstructions)
    return reconstructions, report


def _run_direction(
    config: dict[str, Any],
    model_name: str,
    sign: float,
    target_logits: np.ndarray,
    baseline_logits: np.ndarray,
    records: list[dict[str, Any]],
    extraction: dict[str, np.ndarray],
    reconstructions: dict[str, np.ndarray],
    layer: int,
    rank: int,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    checkpoint = config["checkpoints"][model_name]
    manifest = _load_checkpoint_manifest(config, model_name)
    commit = manifest["checkpoint"]["resolved_commit"]
    cache_dir = Path(config["experiment"]["hf_cache_dir"])
    tokenizer = load_tokenizer(checkpoint["repo_id"], commit, cache_dir)
    model = None
    try:
        model = load_model(
            checkpoint["repo_id"],
            commit,
            cache_dir,
            config["extraction"]["dtype"],
            config["extraction"]["device"],
        )
        strength_results = []
        for strength in config["causal"]["strengths"]:
            logits = _intervened_logits(
                model,
                tokenizer,
                records,
                extraction,
                validation_indices,
                layer,
                reconstructions["pca"],
                float(strength),
                sign,
                int(config["data"]["max_length"]),
                config["extraction"]["device"],
                f"{model_name} validation PCA strength={strength}",
            )
            metrics = normalized_gap_recovery(
                target_logits[validation_indices], baseline_logits[validation_indices], logits
            )
            strength_results.append({"strength": float(strength), "metrics": metrics})
        selected_strength_result = max(
            strength_results, key=lambda item: item["metrics"]["aggregate_recovery"]
        )
        selected_strength = float(selected_strength_result["strength"])

        methods: dict[str, np.ndarray] = {
            "full_delta": reconstructions["delta"],
            "global_pca": reconstructions["pca"],
            "local_pca": reconstructions["local"],
            "autoencoder": reconstructions["autoencoder"],
        }
        for method_name, vectors in reconstructions.items():
            if method_name.startswith("pca_rank_") and int(method_name.rsplit("_", 1)[1]) != rank:
                methods[method_name] = vectors
        for random_index in range(int(config["causal"]["random_controls"])):
            methods[f"random_rotated_{random_index}"] = _random_rotated_vectors(
                reconstructions["pca"],
                reconstructions["mean"],
                reconstructions["basis"],
                int(config["experiment"]["seed"]) + 50_000 + random_index,
            )

        test_results = {}
        for method_name, vectors in methods.items():
            logits = _intervened_logits(
                model,
                tokenizer,
                records,
                extraction,
                test_indices,
                layer,
                vectors,
                selected_strength,
                sign,
                int(config["data"]["max_length"]),
                config["extraction"]["device"],
                f"{model_name} test {method_name}",
            )
            test_results[method_name] = normalized_gap_recovery(
                target_logits[test_indices], baseline_logits[test_indices], logits
            )
        return {
            "model": model_name,
            "sign": sign,
            "selected_strength": selected_strength,
            "validation_strength_sweep": strength_results,
            "test": test_results,
            "layer": layer,
            "rank": rank,
        }
    finally:
        release_model(model)


def run_causal(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    root = artifact_dir(config) / "causal"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    if report_path.exists() and not force:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        print(f"Using existing causal report: {report_path}")
        return report

    geometry_path = artifact_dir(config) / "geometry" / "report.json"
    split_path = artifact_dir(config) / "geometry" / "oracle_reconstructions.npz"
    with geometry_path.open("r", encoding="utf-8") as handle:
        geometry_report = json.load(handle)
    with np.load(split_path, allow_pickle=False) as archive:
        split = {
            "train": archive["train_indices"].astype(np.int64),
            "validation": archive["validation_indices"].astype(np.int64),
            "test": archive["test_indices"].astype(np.int64),
        }

    parent = load_extraction(config, "sft")
    donor = load_extraction(config, "dpo")
    records = load_jsonl(config["data"]["path"])
    all_layer_delta = (
        donor["activations"].astype(np.float32) - parent["activations"].astype(np.float32)
    )
    screen_indices = split["validation"][: int(config["causal"]["layer_screen_examples"])]
    addition_screen = _screen_direction(
        config,
        "sft",
        +1.0,
        donor["logits"].astype(np.float32),
        parent["logits"].astype(np.float32),
        records,
        parent,
        all_layer_delta,
        screen_indices,
    )
    removal_screen = _screen_direction(
        config,
        "dpo",
        -1.0,
        parent["logits"].astype(np.float32),
        donor["logits"].astype(np.float32),
        records,
        donor,
        all_layer_delta,
        screen_indices,
    )
    layer_screen = []
    for add, remove in zip(addition_screen, removal_screen):
        if add["layer"] != remove["layer"]:
            raise AssertionError("Addition/removal layer screens are misaligned")
        layer_screen.append(
            {
                "layer": add["layer"],
                "addition_full_delta_recovery": add["full_delta"]["aggregate_recovery"],
                "removal_full_delta_recovery": remove["full_delta"]["aggregate_recovery"],
                "mean_bidirectional_recovery": 0.5
                * (
                    add["full_delta"]["aggregate_recovery"]
                    + remove["full_delta"]["aggregate_recovery"]
                ),
            }
        )
    selected_screen = max(layer_screen, key=lambda item: item["mean_bidirectional_recovery"])
    layer = int(selected_screen["layer"])
    reconstructions, causal_layer_geometry = _fit_causal_layer_geometry(
        config, geometry_report, all_layer_delta, layer, split
    )
    rank = int(causal_layer_geometry["rank"])
    reconstruction_output = root / "oracle_reconstructions.npz"
    np.savez(reconstruction_output, **reconstructions)

    validation_indices = reconstructions["validation_indices"].astype(np.int64)
    test_indices = reconstructions["test_indices"].astype(np.int64)
    validation_indices = validation_indices[: int(config["causal"]["max_validation_examples"])]
    test_indices = test_indices[: int(config["causal"]["max_test_examples"])]
    started = time.time()

    addition = _run_direction(
        config,
        "sft",
        +1.0,
        donor["logits"].astype(np.float32),
        parent["logits"].astype(np.float32),
        records,
        parent,
        reconstructions,
        layer,
        rank,
        validation_indices,
        test_indices,
    )
    removal = _run_direction(
        config,
        "dpo",
        -1.0,
        parent["logits"].astype(np.float32),
        donor["logits"].astype(np.float32),
        records,
        donor,
        reconstructions,
        layer,
        rank,
        validation_indices,
        test_indices,
    )

    pca_recovery = 0.5 * (
        addition["test"]["global_pca"]["aggregate_recovery"]
        + removal["test"]["global_pca"]["aggregate_recovery"]
    )
    nonlinear_recovery = max(
        0.5
        * (
            addition["test"][method]["aggregate_recovery"]
            + removal["test"][method]["aggregate_recovery"]
        )
        for method in ("local_pca", "autoencoder")
    )
    random_recoveries = [
        0.5
        * (
            addition["test"][name]["aggregate_recovery"]
            + removal["test"][name]["aggregate_recovery"]
        )
        for name in addition["test"]
        if name.startswith("random_rotated_")
    ]
    compact = bool(causal_layer_geometry["compact_candidate"])
    causal = bool(pca_recovery >= 0.20 and pca_recovery > max(random_recoveries))
    nonlinear_gain = float(nonlinear_recovery - pca_recovery)
    rank_curve = []
    for geometry_point in causal_layer_geometry["rank_curve_geometry"]:
        curve_rank = int(geometry_point["rank"])
        method_name = "global_pca" if curve_rank == rank else f"pca_rank_{curve_rank}"
        addition_recovery = addition["test"][method_name]["aggregate_recovery"]
        removal_recovery = removal["test"][method_name]["aggregate_recovery"]
        rank_curve.append(
            {
                "rank": curve_rank,
                "rank_fraction_of_width": curve_rank / all_layer_delta.shape[2],
                "test_centered_r2": geometry_point["test"]["r2_about_train_mean"],
                "test_raw_energy_fraction": geometry_point["test"][
                    "fraction_raw_energy_reconstructed"
                ],
                "addition_recovery": addition_recovery,
                "removal_recovery": removal_recovery,
                "mean_bidirectional_recovery": 0.5
                * (addition_recovery + removal_recovery),
                "paired_prompt_bootstrap": paired_bidirectional_bootstrap(
                    addition["test"][method_name],
                    removal["test"][method_name],
                    int(config["experiment"]["seed"]) + 70_000 + curve_rank,
                ),
            }
        )
    if compact and causal and nonlinear_gain >= 0.10:
        classification = "nonlinear_manifold_candidate"
    elif compact and causal:
        classification = "global_low_rank_subspace_candidate"
    elif causal:
        classification = "causally_concentrated_subspace_partial_delta"
    elif compact:
        classification = "compact_readout_without_causal_recovery"
    else:
        classification = "compact_geometry_not_supported_in_smoke_test"

    report = {
        "kind": "single_layer_bidirectional_causal_test",
        "causal_layer_screen": {
            "examples": len(screen_indices),
            "strength": float(config["causal"]["layer_screen_strength"]),
            "selected": selected_screen,
            "all_layers": layer_screen,
        },
        "causal_layer_geometry": causal_layer_geometry,
        "causal_rank_curve": rank_curve,
        "addition_sft_to_dpo": addition,
        "removal_dpo_to_sft": removal,
        "decision": {
            "classification": classification,
            "compact_reconstruction": compact,
            "mean_bidirectional_pca_recovery": pca_recovery,
            "mean_bidirectional_best_nonlinear_recovery": nonlinear_recovery,
            "nonlinear_causal_gain": nonlinear_gain,
            "random_control_recoveries": random_recoveries,
            "causal_threshold_passed": causal,
            "confirmatory_status": "exploratory_smoke_only",
        },
        "examples": {
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "elapsed_seconds": time.time() - started,
        "runtime": runtime_manifest(),
    }
    write_json(report, report_path)
    print(f"Causal classification: {classification}")
    print(f"Wrote causal report: {report_path}")
    return report
