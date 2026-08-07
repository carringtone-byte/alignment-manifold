from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alignment_manifold.config import artifact_dir
from alignment_manifold.extract import assert_matched_extractions, load_extraction
from alignment_manifold.geometry import (
    _bootstrap_stability,
    _fit_autoencoder,
    _fit_pca,
    _project,
    _projection_similarity,
    _random_subspace_metrics,
    _split_indices,
    reconstruction_metrics,
)
from alignment_manifold.provenance import runtime_manifest, write_json


def _pooled(values: dict[str, np.ndarray], names: list[str], indices: np.ndarray) -> np.ndarray:
    return np.concatenate([values[name][indices] for name in names], axis=0)


def _decode_latent_interpolation(
    model: Any,
    left: np.ndarray,
    right: np.ndarray,
    weight: float,
    mean: np.ndarray,
    scale: float,
    device: str,
) -> np.ndarray:
    left_tensor = torch.from_numpy((left - mean) / scale).float().to(device)
    right_tensor = torch.from_numpy((right - mean) / scale).float().to(device)
    with torch.inference_mode():
        left_latent = model.encoder(left_tensor)
        right_latent = model.encoder(right_tensor)
        latent = (1.0 - weight) * left_latent + weight * right_latent
        decoded = model.decoder(latent).cpu().numpy()
    return mean + scale * decoded


def _interpolation_weight(left_step: float, middle_step: float, right_step: float) -> float:
    if not left_step < middle_step < right_step:
        raise ValueError(
            "Interpolation checkpoints must have strictly increasing training steps"
        )
    return (middle_step - left_step) / (right_step - left_step)


def run_trajectory_geometry(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    root = artifact_dir(config) / "trajectory_geometry"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    if report_path.exists() and not force:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        print(f"Using existing trajectory report: {report_path}")
        return report

    trajectory_config = config["trajectory"]
    reference_name = trajectory_config["reference_checkpoint"]
    checkpoint_names = list(trajectory_config["ordered_checkpoints"])
    reference = load_extraction(config, reference_name)
    extractions = {}
    for name in checkpoint_names:
        extraction = load_extraction(config, name)
        assert_matched_extractions(reference, extraction)
        extractions[name] = extraction

    all_layer_deltas = {
        name: extraction["activations"].astype(np.float32)
        - reference["activations"].astype(np.float32)
        for name, extraction in extractions.items()
    }
    seed = int(config["experiment"]["seed"])
    split = _split_indices(
        reference["cluster_ids"],
        float(config["data"]["train_fraction"]),
        float(config["data"]["validation_fraction"]),
        seed,
    )
    ranks = [int(value) for value in config["geometry"]["ranks"]]
    started = time.time()
    layer_rank_results = []

    for layer in range(reference["activations"].shape[1]):
        layer_values = {name: values[:, layer] for name, values in all_layer_deltas.items()}
        train = _pooled(layer_values, checkpoint_names, split["train"])
        validation = _pooled(layer_values, checkpoint_names, split["validation"])
        test = _pooled(layer_values, checkpoint_names, split["test"])
        for rank in ranks:
            mean, basis = _fit_pca(train, rank, seed + layer * 100 + rank)
            validation_prediction = _project(validation, mean, basis)
            test_prediction = _project(test, mean, basis)
            layer_rank_results.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "validation": reconstruction_metrics(
                        validation, validation_prediction, mean
                    ),
                    "test": reconstruction_metrics(test, test_prediction, mean),
                }
            )
        print(f"trajectory geometry: fitted layer {layer + 1}/{reference['activations'].shape[1]}")

    layer = int(trajectory_config["primary_layer"])
    layer_values = {name: values[:, layer] for name, values in all_layer_deltas.items()}
    train = _pooled(layer_values, checkpoint_names, split["train"])
    validation = _pooled(layer_values, checkpoint_names, split["validation"])
    test = _pooled(layer_values, checkpoint_names, split["test"])
    primary_rows = [row for row in layer_rank_results if row["layer"] == layer]
    best_validation = max(row["validation"]["r2_about_train_mean"] for row in primary_rows)
    near_best = [
        row
        for row in primary_rows
        if row["validation"]["r2_about_train_mean"] >= best_validation - 0.01
    ]
    selected_row = sorted(
        near_best,
        key=lambda row: (row["rank"], -row["validation"]["r2_about_train_mean"]),
    )[0]
    rank = int(selected_row["rank"])
    mean, basis = _fit_pca(train, rank, seed + 80_000 + layer)
    global_predictions = {
        name: _project(values, mean, basis).astype(np.float32)
        for name, values in layer_values.items()
    }
    global_validation = _pooled(global_predictions, checkpoint_names, split["validation"])
    global_test = _pooled(global_predictions, checkpoint_names, split["test"])

    local_predictions = {}
    local_bases = {}
    local_means = {}
    for checkpoint_index, name in enumerate(checkpoint_names):
        local_mean, local_basis = _fit_pca(
            layer_values[name][split["train"]],
            rank,
            seed + 81_000 + checkpoint_index,
        )
        local_means[name] = local_mean
        local_bases[name] = local_basis
        local_predictions[name] = _project(
            layer_values[name], local_mean, local_basis
        ).astype(np.float32)
    local_validation = _pooled(local_predictions, checkpoint_names, split["validation"])
    local_test = _pooled(local_predictions, checkpoint_names, split["test"])

    ae_device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder, ae_mean, ae_scale, ae_training = _fit_autoencoder(
        train, validation, rank, seed + 82_000, ae_device
    )
    autoencoder_predictions = {}
    with torch.inference_mode():
        for name, values in layer_values.items():
            normalized = torch.from_numpy((values - ae_mean) / ae_scale).float().to(
                ae_device
            )
            reconstructed = autoencoder(normalized).cpu().numpy()
            autoencoder_predictions[name] = (
                ae_mean + ae_scale * reconstructed
            ).astype(np.float32)
    autoencoder_validation = _pooled(
        autoencoder_predictions, checkpoint_names, split["validation"]
    )
    autoencoder_test = _pooled(autoencoder_predictions, checkpoint_names, split["test"])

    similarity_matrix = []
    for left_name in checkpoint_names:
        similarity_matrix.append(
            [
                _projection_similarity(local_bases[left_name], local_bases[right_name])
                for right_name in checkpoint_names
            ]
        )

    interpolation = trajectory_config.get("interpolation")
    interpolation_enabled = bool(interpolation and interpolation.get("enabled", True))
    interpolation_metrics = None
    latent_interpolation_gain = None
    test_indices = split["test"]
    if interpolation_enabled:
        left_name = interpolation["left"]
        middle_name = interpolation["middle"]
        right_name = interpolation["right"]
        steps = trajectory_config["training_steps"]
        weight = _interpolation_weight(
            float(steps[left_name]),
            float(steps[middle_name]),
            float(steps[right_name]),
        )
        middle_test = layer_values[middle_name][test_indices]
        middle_train_mean = layer_values[middle_name][split["train"]].mean(
            axis=0, dtype=np.float64
        ).astype(np.float32)
        raw_chord = (
            (1.0 - weight) * layer_values[left_name][test_indices]
            + weight * layer_values[right_name][test_indices]
        )
        pca_chord = (
            (1.0 - weight) * global_predictions[left_name][test_indices]
            + weight * global_predictions[right_name][test_indices]
        )
        latent_chord = _decode_latent_interpolation(
            autoencoder,
            layer_values[left_name][test_indices],
            layer_values[right_name][test_indices],
            weight,
            ae_mean,
            ae_scale,
            ae_device,
        )

    autoencoder.to("cpu")
    del autoencoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    global_metrics = reconstruction_metrics(test, global_test, mean)
    local_metrics = reconstruction_metrics(test, local_test, mean)
    autoencoder_metrics = reconstruction_metrics(test, autoencoder_test, mean)
    if interpolation_enabled:
        interpolation_metrics = {
            "weight": weight,
            "raw_activation_chord": reconstruction_metrics(
                middle_test, raw_chord, middle_train_mean
            ),
            "global_pca_chord": reconstruction_metrics(
                middle_test, pca_chord, middle_train_mean
            ),
            "autoencoder_latent_chord": reconstruction_metrics(
                middle_test, latent_chord, middle_train_mean
            ),
        }
    nonlinear_gain = autoencoder_metrics["r2_about_train_mean"] - global_metrics[
        "r2_about_train_mean"
    ]
    local_gain = local_metrics["r2_about_train_mean"] - global_metrics[
        "r2_about_train_mean"
    ]
    if interpolation_metrics:
        latent_interpolation_gain = interpolation_metrics["autoencoder_latent_chord"][
            "r2_about_train_mean"
        ] - interpolation_metrics["raw_activation_chord"]["r2_about_train_mean"]
    if nonlinear_gain >= 0.05 and latent_interpolation_gain is not None and latent_interpolation_gain >= 0.05:
        classification = "nonlinear_trajectory_manifold_candidate"
    elif local_gain >= 0.05:
        classification = "checkpoint_varying_union_of_subspaces_candidate"
    elif global_metrics["r2_about_train_mean"] >= 0.70:
        classification = "shared_global_trajectory_subspace_candidate"
    else:
        classification = "compact_shared_trajectory_geometry_not_supported"

    per_checkpoint = {}
    for name in checkpoint_names:
        values = layer_values[name][test_indices]
        per_checkpoint[name] = {
            "global_pca": reconstruction_metrics(
                values, global_predictions[name][test_indices], mean
            ),
            "checkpoint_local_pca": reconstruction_metrics(
                values, local_predictions[name][test_indices], mean
            ),
            "autoencoder": reconstruction_metrics(
                values, autoencoder_predictions[name][test_indices], mean
            ),
        }

    report = {
        "kind": "multi_checkpoint_activation_trajectory_geometry",
        "reference_checkpoint": reference_name,
        "checkpoint_order": checkpoint_names,
        "training_steps": trajectory_config["training_steps"],
        "primary_layer": layer,
        "selected_rank": rank,
        "selection_rule": "smallest rank within 0.01 pooled validation R2 of the primary layer maximum",
        "global_pca": {
            "validation": reconstruction_metrics(validation, global_validation, mean),
            "test": global_metrics,
            "random_test_r2": _random_subspace_metrics(
                train,
                test,
                rank,
                int(config["geometry"]["random_controls"]),
                seed + 83_000,
            ),
            "bootstrap_stability": _bootstrap_stability(
                train,
                basis,
                rank,
                int(config["geometry"]["bootstrap_samples"]),
                seed + 84_000,
            ),
        },
        "checkpoint_local_pca": {
            "test": local_metrics,
            "gain_over_global": local_gain,
            "subspace_similarity_names": checkpoint_names,
            "subspace_similarity_matrix": similarity_matrix,
        },
        "autoencoder": {
            "training": ae_training,
            "test": autoencoder_metrics,
            "gain_over_global": nonlinear_gain,
        },
        "interpolation": interpolation_metrics,
        "latent_interpolation_gain_over_raw_chord": latent_interpolation_gain,
        "per_checkpoint_test": per_checkpoint,
        "layer_rank_results": layer_rank_results,
        "decision": {
            "classification": classification,
            "confirmatory_status": "exploratory_multi_checkpoint_smoke",
        },
        "splits": {
            name: {
                "examples": int(len(indices)),
                "clusters": int(len(set(reference["cluster_ids"][indices].tolist()))),
            }
            for name, indices in split.items()
        },
        "elapsed_seconds": time.time() - started,
        "runtime": runtime_manifest(),
    }
    write_json(report, report_path)
    print(f"Trajectory classification: {classification}")
    print(f"Wrote trajectory report: {report_path}")
    return report
