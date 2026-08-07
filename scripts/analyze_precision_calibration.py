from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alignment_manifold.provenance import runtime_manifest, write_json  # noqa: E402


def vector_metrics(target: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    target_flat = target.reshape(-1, target.shape[-1]).astype(np.float64)
    candidate_flat = candidate.reshape(-1, candidate.shape[-1]).astype(np.float64)
    error = candidate_flat - target_flat
    target_norm = np.linalg.norm(target_flat, axis=1)
    candidate_norm = np.linalg.norm(candidate_flat, axis=1)
    error_norm = np.linalg.norm(error, axis=1)
    cosine = np.sum(target_flat * candidate_flat, axis=1) / np.maximum(
        target_norm * candidate_norm, 1e-12
    )
    target_mean = target_flat.mean(axis=0)
    centered_denominator = np.square(target_flat - target_mean).sum()
    r2 = (
        1.0 - np.square(error).sum() / centered_denominator
        if centered_denominator > 0
        else float("nan")
    )
    return {
        "mean_cosine": float(cosine.mean()),
        "minimum_cosine": float(cosine.min()),
        "mean_relative_error": float(np.mean(error_norm / np.maximum(target_norm, 1e-12))),
        "median_relative_error": float(np.median(error_norm / np.maximum(target_norm, 1e-12))),
        "mean_norm_ratio_nf4_over_fp16": float(
            np.mean(candidate_norm / np.maximum(target_norm, 1e-12))
        ),
        "r2_nf4_predicting_fp16_about_fp16_mean": float(r2),
    }


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp16-dir", default="artifacts/trajectory_7b_fp16_calibration/extractions"
    )
    parser.add_argument("--nf4-dir", default="artifacts/trajectory_7b/extractions")
    parser.add_argument(
        "--output", default="artifacts/trajectory_7b_fp16_calibration/precision_report.json"
    )
    args = parser.parse_args()
    fp16_dir = ROOT / args.fp16_dir
    nf4_dir = ROOT / args.nf4_dir
    fp16 = {name: load(fp16_dir / f"{name}.npz") for name in ("dpo", "rlvr_0180")}
    nf4_full = {name: load(nf4_dir / f"{name}.npz") for name in ("dpo", "rlvr_0180")}
    nf4 = {}
    for name in fp16:
        lookup = {
            str(example_id): index
            for index, example_id in enumerate(nf4_full[name]["example_ids"].tolist())
        }
        indices = np.asarray([lookup[str(value)] for value in fp16[name]["example_ids"]])
        nf4[name] = nf4_full[name]["activations"][indices].astype(np.float32)
        if not np.array_equal(
            fp16[name]["example_ids"], nf4_full[name]["example_ids"][indices]
        ):
            raise AssertionError("Precision calibration example order mismatch")

    fp16_activations = {
        name: fp16[name]["activations"].astype(np.float32) for name in fp16
    }
    fp16_delta = fp16_activations["rlvr_0180"] - fp16_activations["dpo"]
    nf4_delta = nf4["rlvr_0180"] - nf4["dpo"]
    layers = []
    for layer in range(fp16_delta.shape[1]):
        layers.append(
            {
                "layer": layer,
                "dpo_activation": vector_metrics(
                    fp16_activations["dpo"][:, layer], nf4["dpo"][:, layer]
                ),
                "rlvr_0180_activation": vector_metrics(
                    fp16_activations["rlvr_0180"][:, layer], nf4["rlvr_0180"][:, layer]
                ),
                "checkpoint_delta": vector_metrics(
                    fp16_delta[:, layer], nf4_delta[:, layer]
                ),
            }
        )
    report = {
        "kind": "nf4_against_fp16_activation_calibration",
        "status": "exploratory_followup",
        "examples": int(fp16_delta.shape[0]),
        "layers": int(fp16_delta.shape[1]),
        "hidden_size": int(fp16_delta.shape[2]),
        "checkpoints": ["dpo", "rlvr_0180"],
        "primary_layer": 28,
        "primary_layer_metrics": layers[28],
        "all_layers": layers,
        "runtime": runtime_manifest(),
    }
    output = ROOT / args.output
    write_json(report, output)
    print(json.dumps({key: value for key, value in report.items() if key != "all_layers"}, indent=2))
    print(f"Wrote precision report: {output}")


if __name__ == "__main__":
    main()
