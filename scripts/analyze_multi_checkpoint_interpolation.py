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

from alignment_manifold.config import load_config  # noqa: E402
from alignment_manifold.geometry import (  # noqa: E402
    _fit_pca,
    _project,
    _split_indices,
    reconstruction_metrics,
)
from alignment_manifold.provenance import runtime_manifest, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trajectory_7b_expanded_six.yaml")
    parser.add_argument(
        "--output", default="artifacts/trajectory_7b_expanded_six/multi_interpolation/report.json"
    )
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    extraction_dir = ROOT / config["experiment"]["artifact_dir"] / "extractions"
    reference_name = config["trajectory"]["reference_checkpoint"]
    names = list(config["trajectory"]["ordered_checkpoints"])
    steps = config["trajectory"]["training_steps"]
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
    train = np.concatenate([values[name][split["train"]] for name in names], axis=0)
    global_mean, global_basis = _fit_pca(
        train, args.rank, int(config["experiment"]["seed"]) + 80_000 + args.layer
    )
    projected = {name: _project(values[name], global_mean, global_basis) for name in names}
    results = []
    for index in range(1, len(names) - 1):
        left, middle, right = names[index - 1 : index + 2]
        weight = (steps[middle] - steps[left]) / (steps[right] - steps[left])
        target = values[middle][split["test"]]
        target_mean = values[middle][split["train"]].mean(
            axis=0, dtype=np.float64
        ).astype(np.float32)
        raw = (1.0 - weight) * values[left][split["test"]] + weight * values[right][
            split["test"]
        ]
        pca = (1.0 - weight) * projected[left][split["test"]] + weight * projected[
            right
        ][split["test"]]
        results.append(
            {
                "left": left,
                "middle": middle,
                "right": right,
                "steps": [steps[left], steps[middle], steps[right]],
                "weight": weight,
                "raw_chord": reconstruction_metrics(target, raw, target_mean),
                "global_pca_chord": reconstruction_metrics(target, pca, target_mean),
            }
        )
    raw_scores = np.asarray([row["raw_chord"]["r2_about_train_mean"] for row in results])
    pca_scores = np.asarray(
        [row["global_pca_chord"]["r2_about_train_mean"] for row in results]
    )
    report = {
        "kind": "adjacent_multi_checkpoint_interpolation",
        "status": "exploratory_followup",
        "layer": args.layer,
        "rank": args.rank,
        "checkpoint_order": names,
        "results": results,
        "summary": {
            "raw_chord_mean_r2": float(raw_scores.mean()),
            "raw_chord_minimum_r2": float(raw_scores.min()),
            "raw_chord_maximum_r2": float(raw_scores.max()),
            "global_pca_chord_mean_r2": float(pca_scores.mean()),
            "global_pca_chord_minimum_r2": float(pca_scores.min()),
            "global_pca_chord_maximum_r2": float(pca_scores.max()),
        },
        "runtime": runtime_manifest(),
    }
    output = ROOT / args.output
    write_json(report, output)
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote multi-checkpoint interpolation report: {output}")


if __name__ == "__main__":
    main()
