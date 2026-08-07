from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

from alignment_manifold.config import artifact_dir
from alignment_manifold.extract import assert_matched_extractions, load_extraction
from alignment_manifold.provenance import runtime_manifest, write_json


def _split_indices(
    groups: np.ndarray, train_fraction: float, validation_fraction: float, seed: int
) -> dict[str, np.ndarray]:
    indices = np.arange(len(groups))
    first = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    train_indices, remainder_indices = next(first.split(indices, groups=groups))
    relative_validation = validation_fraction / (1.0 - train_fraction)
    second = GroupShuffleSplit(
        n_splits=1, train_size=relative_validation, random_state=seed + 1
    )
    val_relative, test_relative = next(
        second.split(remainder_indices, groups=groups[remainder_indices])
    )
    result = {
        "train": np.sort(train_indices),
        "validation": np.sort(remainder_indices[val_relative]),
        "test": np.sort(remainder_indices[test_relative]),
    }
    group_sets = {name: set(groups[idx].tolist()) for name, idx in result.items()}
    if group_sets["train"] & group_sets["validation"]:
        raise AssertionError("Train and validation groups overlap")
    if group_sets["train"] & group_sets["test"]:
        raise AssertionError("Train and test groups overlap")
    if group_sets["validation"] & group_sets["test"]:
        raise AssertionError("Validation and test groups overlap")
    return result


def reconstruction_r2(values: np.ndarray, predictions: np.ndarray, mean: np.ndarray) -> float:
    numerator = float(np.square(values - predictions, dtype=np.float64).sum())
    denominator = float(np.square(values - mean, dtype=np.float64).sum())
    if denominator <= 0:
        return float("nan")
    return 1.0 - numerator / denominator


def reconstruction_metrics(
    values: np.ndarray, predictions: np.ndarray, train_mean: np.ndarray
) -> dict[str, float]:
    error = values - predictions
    numerator = float(np.square(error, dtype=np.float64).sum())
    value_energy = float(np.square(values, dtype=np.float64).sum())
    centered_energy = float(np.square(values - train_mean, dtype=np.float64).sum())
    flat_values = values.reshape(len(values), -1).astype(np.float64)
    flat_predictions = predictions.reshape(len(predictions), -1).astype(np.float64)
    dot = np.sum(flat_values * flat_predictions, axis=1)
    norm = np.linalg.norm(flat_values, axis=1) * np.linalg.norm(flat_predictions, axis=1)
    cosine = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
    return {
        "r2_about_train_mean": 1.0 - numerator / centered_energy if centered_energy > 0 else float("nan"),
        "fraction_raw_energy_reconstructed": 1.0 - numerator / value_energy if value_energy > 0 else float("nan"),
        "normalized_frobenius_error": math.sqrt(numerator / value_energy) if value_energy > 0 else float("nan"),
        "mean_reconstruction_cosine": float(np.mean(cosine)),
    }


def _fit_pca(train: np.ndarray, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    effective_rank = min(rank, len(train) - 1, train.shape[1])
    if effective_rank < 1:
        raise ValueError("Not enough training samples for PCA")
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    pca = PCA(n_components=effective_rank, svd_solver="randomized", random_state=seed)
    pca.fit(train - mean)
    return mean, pca.components_.astype(np.float32)


def _project(values: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = values - mean
    return mean + (centered @ basis.T) @ basis


def _random_subspace_metrics(
    train: np.ndarray,
    values: np.ndarray,
    rank: int,
    count: int,
    seed: int,
) -> dict[str, float]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(count):
        matrix = rng.standard_normal((train.shape[1], rank), dtype=np.float32)
        basis, _ = np.linalg.qr(matrix, mode="reduced")
        prediction = _project(values, mean, basis.T)
        scores.append(reconstruction_r2(values, prediction, mean))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "maximum": float(np.max(scores)),
    }


def _projection_similarity(left: np.ndarray, right: np.ndarray) -> float:
    rank = min(len(left), len(right))
    return float(np.square(left[:rank] @ right[:rank].T, dtype=np.float64).sum() / rank)


def _bootstrap_stability(
    train: np.ndarray, basis: np.ndarray, rank: int, samples: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for bootstrap_index in range(samples):
        indices = rng.integers(0, len(train), size=len(train))
        _, bootstrap_basis = _fit_pca(train[indices], rank, seed + bootstrap_index + 1)
        scores.append(_projection_similarity(basis, bootstrap_basis))
    return {
        "mean_projection_similarity": float(np.mean(scores)),
        "ci_025": float(np.quantile(scores, 0.025)),
        "ci_975": float(np.quantile(scores, 0.975)),
    }


def _fit_local_pca(
    train: np.ndarray, rank: int, components: int, seed: int
) -> dict[str, np.ndarray]:
    kmeans = KMeans(n_clusters=components, n_init=20, random_state=seed)
    labels = kmeans.fit_predict(train)
    bases = np.zeros((components, rank, train.shape[1]), dtype=np.float32)
    local_ranks = np.zeros(components, dtype=np.int32)
    means = np.zeros((components, train.shape[1]), dtype=np.float32)
    for cluster in range(components):
        cluster_values = train[labels == cluster]
        local_rank = min(rank, len(cluster_values) - 1, train.shape[1])
        if local_rank < 1:
            means[cluster] = kmeans.cluster_centers_[cluster]
            continue
        mean, basis = _fit_pca(cluster_values, local_rank, seed + cluster + 1)
        means[cluster] = mean
        bases[cluster, :local_rank] = basis
        local_ranks[cluster] = local_rank
    return {
        "assignment_centers": kmeans.cluster_centers_.astype(np.float32),
        "means": means,
        "bases": bases,
        "local_ranks": local_ranks,
    }


def _reconstruct_local(values: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    centers = model["assignment_centers"]
    distances = np.square(values[:, None, :] - centers[None, :, :], dtype=np.float64).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    predictions = np.empty_like(values, dtype=np.float32)
    for index, cluster in enumerate(labels):
        rank = int(model["local_ranks"][cluster])
        mean = model["means"][cluster]
        if rank == 0:
            predictions[index] = mean
        else:
            basis = model["bases"][cluster, :rank]
            predictions[index] = _project(values[index : index + 1], mean, basis)[0]
    return predictions


class DeltaAutoencoder(torch.nn.Module):
    def __init__(self, width: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, latent),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def _fit_autoencoder(
    train: np.ndarray,
    validation: np.ndarray,
    rank: int,
    seed: int,
    device: str,
    hidden: int = 64,
    max_epochs: int = 1000,
    patience: int = 75,
) -> tuple[DeltaAutoencoder, np.ndarray, float, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = float(np.sqrt(np.square(train - mean, dtype=np.float64).mean()))
    scale = max(scale, 1e-8)
    train_tensor = torch.from_numpy((train - mean) / scale).float().to(device)
    validation_tensor = torch.from_numpy((validation - mean) / scale).float().to(device)
    model = DeltaAutoencoder(train.shape[1], hidden, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale = 0
    best_epoch = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_tensor), train_tensor)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                torch.nn.functional.mse_loss(model(validation_tensor), validation_tensor).item()
            )
        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, scale, {
        "hidden_width": hidden,
        "best_epoch": best_epoch,
        "epochs_ran": epoch + 1,
        "best_validation_mse_normalized": best_validation,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _reconstruct_autoencoder(
    model: DeltaAutoencoder,
    values: np.ndarray,
    mean: np.ndarray,
    scale: float,
    device: str,
) -> np.ndarray:
    tensor = torch.from_numpy((values - mean) / scale).float().to(device)
    with torch.inference_mode():
        prediction = model(tensor).cpu().numpy()
    return mean + scale * prediction


def run_geometry(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    root = artifact_dir(config) / "geometry"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    reconstruction_path = root / "oracle_reconstructions.npz"
    if report_path.exists() and reconstruction_path.exists() and not force:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        print(f"Using existing geometry report: {report_path}")
        return report

    parent = load_extraction(config, "sft")
    donor = load_extraction(config, "dpo")
    assert_matched_extractions(parent, donor)
    delta = donor["activations"].astype(np.float32) - parent["activations"].astype(np.float32)
    seed = int(config["experiment"]["seed"])
    split = _split_indices(
        parent["cluster_ids"],
        float(config["data"]["train_fraction"]),
        float(config["data"]["validation_fraction"]),
        seed,
    )
    ranks = [int(rank) for rank in config["geometry"]["ranks"]]
    layer_results: list[dict[str, Any]] = []
    fitted: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    started = time.time()

    for layer in range(delta.shape[1]):
        train = delta[split["train"], layer]
        validation = delta[split["validation"], layer]
        test = delta[split["test"], layer]
        layer_mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
        mean_prediction = np.broadcast_to(layer_mean, test.shape)
        mean_raw_fraction = reconstruction_metrics(test, mean_prediction, np.zeros_like(layer_mean))[
            "fraction_raw_energy_reconstructed"
        ]
        for rank in ranks:
            mean, basis = _fit_pca(train, rank, seed + layer * 100 + rank)
            fitted[(layer, rank)] = (mean, basis)
            validation_prediction = _project(validation, mean, basis)
            test_prediction = _project(test, mean, basis)
            result = {
                "layer": layer,
                "rank": rank,
                "mean_raw_energy_fraction": mean_raw_fraction,
                "validation": reconstruction_metrics(validation, validation_prediction, mean),
                "test": reconstruction_metrics(test, test_prediction, mean),
                "random_test_r2": _random_subspace_metrics(
                    train,
                    test,
                    rank,
                    int(config["geometry"]["random_controls"]),
                    seed + 10_000 + layer * 100 + rank,
                ),
            }
            layer_results.append(result)
        print(f"geometry: fitted PCA for layer {layer + 1}/{delta.shape[1]}")

    best_validation = max(
        item["validation"]["r2_about_train_mean"] for item in layer_results
    )
    near_best = [
        item
        for item in layer_results
        if item["validation"]["r2_about_train_mean"] >= best_validation - 0.01
    ]
    selected = sorted(
        near_best,
        key=lambda item: (item["rank"], -item["validation"]["r2_about_train_mean"], item["layer"]),
    )[0]
    selected_layer = int(selected["layer"])
    selected_rank = int(selected["rank"])
    mean, basis = fitted[(selected_layer, selected_rank)]
    selected_train = delta[split["train"], selected_layer]
    selected_validation = delta[split["validation"], selected_layer]
    selected_test = delta[split["test"], selected_layer]
    pca_all = _project(delta[:, selected_layer], mean, basis).astype(np.float32)
    bootstrap = _bootstrap_stability(
        selected_train,
        basis,
        selected_rank,
        int(config["geometry"]["bootstrap_samples"]),
        seed + 20_000,
    )

    local_results = []
    local_models: dict[int, dict[str, np.ndarray]] = {}
    for components in config["geometry"]["local_components"]:
        components = int(components)
        local_model = _fit_local_pca(
            selected_train, selected_rank, components, seed + 30_000 + components
        )
        local_models[components] = local_model
        validation_prediction = _reconstruct_local(selected_validation, local_model)
        test_prediction = _reconstruct_local(selected_test, local_model)
        local_results.append(
            {
                "components": components,
                "validation": reconstruction_metrics(selected_validation, validation_prediction, mean),
                "test": reconstruction_metrics(selected_test, test_prediction, mean),
                "parameter_count_approx": int(
                    components * selected_rank * delta.shape[2] + 2 * components * delta.shape[2]
                ),
            }
        )
    selected_local_result = max(
        local_results, key=lambda item: item["validation"]["r2_about_train_mean"]
    )
    selected_local_components = int(selected_local_result["components"])
    local_all = _reconstruct_local(
        delta[:, selected_layer], local_models[selected_local_components]
    ).astype(np.float32)

    ae_device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder, ae_mean, ae_scale, ae_training = _fit_autoencoder(
        selected_train,
        selected_validation,
        selected_rank,
        seed + 40_000,
        ae_device,
    )
    ae_all = _reconstruct_autoencoder(
        autoencoder, delta[:, selected_layer], ae_mean, ae_scale, ae_device
    ).astype(np.float32)
    autoencoder_result = {
        "training": ae_training,
        "validation": reconstruction_metrics(
            selected_validation, ae_all[split["validation"]], mean
        ),
        "test": reconstruction_metrics(selected_test, ae_all[split["test"]], mean),
    }
    autoencoder.to("cpu")
    del autoencoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pca_test_r2 = selected["test"]["r2_about_train_mean"]
    best_nonlinear_test_r2 = max(
        selected_local_result["test"]["r2_about_train_mean"],
        autoencoder_result["test"]["r2_about_train_mean"],
    )
    geometry_decision = {
        "compact_candidate": bool(pca_test_r2 >= 0.70),
        "nonlinear_reconstruction_gain": float(best_nonlinear_test_r2 - pca_test_r2),
        "manifold_candidate_from_reconstruction": bool(
            best_nonlinear_test_r2 >= 0.70 and best_nonlinear_test_r2 - pca_test_r2 >= 0.05
        ),
        "confirmatory_status": "exploratory_smoke_only",
    }

    np.savez(
        reconstruction_path,
        delta=delta[:, selected_layer].astype(np.float16),
        pca=pca_all.astype(np.float16),
        local=local_all.astype(np.float16),
        autoencoder=ae_all.astype(np.float16),
        mean=mean.astype(np.float32),
        basis=basis.astype(np.float32),
        train_indices=split["train"],
        validation_indices=split["validation"],
        test_indices=split["test"],
        example_ids=parent["example_ids"],
    )
    report = {
        "kind": "held_out_alignment_delta_geometry",
        "selected_layer": selected_layer,
        "selected_rank": selected_rank,
        "selection_rule": "smallest rank within 0.01 validation R2 of the global maximum",
        "selected_global_pca": selected,
        "bootstrap_stability": bootstrap,
        "local_models": local_results,
        "selected_local_components": selected_local_components,
        "autoencoder": autoencoder_result,
        "decision": geometry_decision,
        "splits": {
            name: {
                "examples": int(len(indices)),
                "clusters": int(len(set(parent["cluster_ids"][indices].tolist()))),
            }
            for name, indices in split.items()
        },
        "all_layer_rank_results": layer_results,
        "elapsed_seconds": time.time() - started,
        "runtime": runtime_manifest(),
    }
    write_json(report, report_path)
    print(f"Selected layer={selected_layer}, rank={selected_rank}")
    print(f"Wrote geometry report: {report_path}")
    return report

