from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alignment_manifold.config import load_config  # noqa: E402
from alignment_manifold.geometry import (  # noqa: E402
    _fit_pca,
    _project,
    _split_indices,
    reconstruction_metrics,
)
from alignment_manifold.provenance import runtime_manifest, write_json  # noqa: E402


class RegularizedAutoencoder(torch.nn.Module):
    def __init__(
        self,
        width: int,
        hidden: int,
        latent: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        activation_layer: type[torch.nn.Module]
        if activation == "gelu":
            activation_layer = torch.nn.GELU
        elif activation == "tanh":
            activation_layer = torch.nn.Tanh
        else:
            raise ValueError(activation)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            activation_layer(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, latent),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent, hidden),
            activation_layer(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def fit_variant(
    train: np.ndarray,
    validation: np.ndarray,
    specification: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, float, RegularizedAutoencoder, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = max(float(np.sqrt(np.square(train - mean, dtype=np.float64).mean())), 1e-8)
    train_tensor = torch.from_numpy((train - mean) / scale).float().to(device)
    validation_tensor = torch.from_numpy((validation - mean) / scale).float().to(device)
    model = RegularizedAutoencoder(
        train.shape[1],
        int(specification["hidden"]),
        int(specification["latent"]),
        str(specification["activation"]),
        float(specification["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=float(specification["weight_decay"])
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    max_epochs = 750
    patience = 60
    noise = float(specification["noise"])
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        corrupted = train_tensor
        if noise > 0:
            corrupted = train_tensor + noise * torch.randn(
                train_tensor.shape, generator=generator, device=device
            )
        loss = torch.nn.functional.mse_loss(model(corrupted), train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                torch.nn.functional.mse_loss(
                    model(validation_tensor), validation_tensor
                ).item()
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
    return mean, scale, model, {
        "best_epoch": best_epoch,
        "epochs_ran": epoch + 1,
        "best_validation_mse_normalized": best_validation,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def reconstruct(
    model: RegularizedAutoencoder,
    values: np.ndarray,
    mean: np.ndarray,
    scale: float,
    device: str,
) -> np.ndarray:
    with torch.inference_mode():
        tensor = torch.from_numpy((values - mean) / scale).float().to(device)
        return mean + scale * model(tensor).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trajectory_7b_expanded_six.yaml")
    parser.add_argument(
        "--output", default="artifacts/trajectory_7b_expanded_six/nonlinear_sweep/report.json"
    )
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    extraction_dir = ROOT / config["experiment"]["artifact_dir"] / "extractions"
    reference_name = config["trajectory"]["reference_checkpoint"]
    names = list(config["trajectory"]["ordered_checkpoints"])
    with np.load(extraction_dir / f"{reference_name}.npz", allow_pickle=False) as archive:
        reference = archive["activations"][:, args.layer].astype(np.float32)
        cluster_ids = archive["cluster_ids"]
    values = {}
    for name in names:
        with np.load(extraction_dir / f"{name}.npz", allow_pickle=False) as archive:
            values[name] = archive["activations"][:, args.layer].astype(np.float32) - reference
    split = _split_indices(
        cluster_ids,
        float(config["data"]["train_fraction"]),
        float(config["data"]["validation_fraction"]),
        int(config["experiment"]["seed"]),
    )
    pooled = {
        split_name: np.concatenate([values[name][indices] for name in names], axis=0)
        for split_name, indices in split.items()
    }
    pca_mean, pca_basis = _fit_pca(
        pooled["train"], args.rank, int(config["experiment"]["seed"]) + 80_000
    )
    pca_validation = _project(pooled["validation"], pca_mean, pca_basis)
    pca_test = _project(pooled["test"], pca_mean, pca_basis)
    pca_metrics = {
        "validation": reconstruction_metrics(
            pooled["validation"], pca_validation, pca_mean
        ),
        "test": reconstruction_metrics(pooled["test"], pca_test, pca_mean),
    }
    variants = [
        {"hidden": 64, "latent": 32, "activation": "tanh", "dropout": 0.0, "noise": 0.0, "weight_decay": 1e-4},
        {"hidden": 128, "latent": 32, "activation": "gelu", "dropout": 0.0, "noise": 0.0, "weight_decay": 1e-4},
        {"hidden": 256, "latent": 32, "activation": "gelu", "dropout": 0.1, "noise": 0.0, "weight_decay": 1e-3},
        {"hidden": 128, "latent": 32, "activation": "gelu", "dropout": 0.1, "noise": 0.05, "weight_decay": 1e-3},
        {"hidden": 256, "latent": 32, "activation": "gelu", "dropout": 0.1, "noise": 0.05, "weight_decay": 1e-3},
        {"hidden": 256, "latent": 64, "activation": "gelu", "dropout": 0.1, "noise": 0.05, "weight_decay": 1e-3},
        {"hidden": 128, "latent": 64, "activation": "tanh", "dropout": 0.1, "noise": 0.02, "weight_decay": 1e-2},
        {"hidden": 256, "latent": 64, "activation": "gelu", "dropout": 0.2, "noise": 0.10, "weight_decay": 1e-2},
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for index, specification in enumerate(variants):
        mean, scale, model, training = fit_variant(
            pooled["train"],
            pooled["validation"],
            specification,
            int(config["experiment"]["seed"]) + 1000 + index,
            device,
        )
        validation_prediction = reconstruct(
            model, pooled["validation"], mean, scale, device
        )
        test_prediction = reconstruct(model, pooled["test"], mean, scale, device)
        result = {
            "variant": specification,
            "training": training,
            "validation": reconstruction_metrics(
                pooled["validation"], validation_prediction, mean
            ),
            "test": reconstruction_metrics(pooled["test"], test_prediction, mean),
        }
        results.append(result)
        print(
            f"variant {index + 1}/{len(variants)} validation R2="
            f"{result['validation']['r2_about_train_mean']:.4f}"
        )
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected = max(results, key=lambda row: row["validation"]["r2_about_train_mean"])
    report = {
        "kind": "regularized_nonlinear_trajectory_sweep",
        "status": "exploratory_followup",
        "layer": args.layer,
        "rank": args.rank,
        "training_rows": int(len(pooled["train"])),
        "validation_rows": int(len(pooled["validation"])),
        "test_rows": int(len(pooled["test"])),
        "global_pca": pca_metrics,
        "variants": results,
        "selected_by_validation": selected,
        "selected_test_gain_over_pca": float(
            selected["test"]["r2_about_train_mean"]
            - pca_metrics["test"]["r2_about_train_mean"]
        ),
        "runtime": runtime_manifest(),
    }
    output = ROOT / args.output
    write_json(report, output)
    print(json.dumps({
        "global_pca_test": pca_metrics["test"],
        "selected": selected,
        "selected_test_gain_over_pca": report["selected_test_gain_over_pca"],
    }, indent=2))
    print(f"Wrote nonlinear sweep report: {output}")


if __name__ == "__main__":
    main()
