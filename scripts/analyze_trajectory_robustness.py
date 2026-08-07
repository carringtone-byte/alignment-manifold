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

from alignment_manifold.geometry import (  # noqa: E402
    _fit_pca,
    _project,
    reconstruction_metrics,
    reconstruction_r2,
)
from alignment_manifold.provenance import runtime_manifest, write_json  # noqa: E402


def stratified_cluster_split(
    cluster_ids: np.ndarray,
    categories: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    group_category: dict[str, str] = {}
    for group in np.unique(cluster_ids):
        values = np.unique(categories[cluster_ids == group])
        if len(values) != 1:
            raise ValueError(f"Cluster {group} crosses categories: {values.tolist()}")
        group_category[str(group)] = str(values[0])
    split_groups = {"train": [], "validation": [], "test": []}
    for category in sorted(set(group_category.values())):
        groups = np.asarray(
            [group for group, value in group_category.items() if value == category]
        )
        groups = groups[rng.permutation(len(groups))]
        train_count = int(round(0.6 * len(groups)))
        validation_count = int(round(0.2 * len(groups)))
        split_groups["train"].extend(groups[:train_count].tolist())
        split_groups["validation"].extend(
            groups[train_count : train_count + validation_count].tolist()
        )
        split_groups["test"].extend(
            groups[train_count + validation_count :].tolist()
        )
    result = {
        name: np.flatnonzero(np.isin(cluster_ids, groups))
        for name, groups in split_groups.items()
    }
    sets = {name: set(cluster_ids[idx].tolist()) for name, idx in result.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets[
        "validation"
    ] & sets["test"]:
        raise AssertionError("Stratified cluster splits overlap")
    return result


def pooled(values: dict[str, np.ndarray], names: list[str], indices: np.ndarray) -> np.ndarray:
    return np.concatenate([values[name][indices] for name in names], axis=0)


def cluster_bootstrap(
    actual: np.ndarray,
    prediction: np.ndarray,
    mean: np.ndarray,
    row_clusters: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    clusters = np.unique(row_clusters)
    sufficient = []
    for cluster in clusters:
        mask = row_clusters == cluster
        numerator = np.square(actual[mask] - prediction[mask], dtype=np.float64).sum()
        denominator = np.square(actual[mask] - mean, dtype=np.float64).sum()
        sufficient.append((numerator, denominator))
    sufficient_array = np.asarray(sufficient)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(samples, len(clusters)))
    sums = sufficient_array[draws].sum(axis=1)
    estimates = 1.0 - sums[:, 0] / sums[:, 1]
    return {
        "mean": float(estimates.mean()),
        "ci_025": float(np.quantile(estimates, 0.025)),
        "median": float(np.quantile(estimates, 0.5)),
        "ci_975": float(np.quantile(estimates, 0.975)),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/trajectory_7b")
    parser.add_argument("--output", default="artifacts/trajectory_7b/robustness/report.json")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()

    extraction_dir = ROOT / args.artifact_dir / "extractions"
    names = ["rlvr_0060", "rlvr_0180", "rlvr_0360"]
    with np.load(extraction_dir / "dpo.npz", allow_pickle=False) as archive:
        reference = archive["activations"][:, args.layer].astype(np.float32)
        cluster_ids = archive["cluster_ids"]
        categories = archive["categories"]
    values = {}
    for name in names:
        with np.load(extraction_dir / f"{name}.npz", allow_pickle=False) as archive:
            values[name] = archive["activations"][:, args.layer].astype(np.float32) - reference

    seed_results = []
    for seed_offset in range(args.seeds):
        seed = 1729 + seed_offset
        split = stratified_cluster_split(cluster_ids, categories, seed)
        train = pooled(values, names, split["train"])
        validation = pooled(values, names, split["validation"])
        test = pooled(values, names, split["test"])
        mean, basis = _fit_pca(train, args.rank, seed + 80_000 + args.layer)
        validation_prediction = _project(validation, mean, basis)
        test_prediction = _project(test, mean, basis)
        category_results = {}
        for category in sorted(np.unique(categories).tolist()):
            indices = split["test"][categories[split["test"]] == category]
            actual = pooled(values, names, indices)
            prediction = _project(actual, mean, basis)
            category_results[str(category)] = {
                "prompts": int(len(indices)),
                **reconstruction_metrics(actual, prediction, mean),
            }
        middle_mean = values["rlvr_0180"][split["train"]].mean(
            axis=0, dtype=np.float64
        ).astype(np.float32)
        middle = values["rlvr_0180"][split["test"]]
        raw_chord = (
            0.6 * values["rlvr_0060"][split["test"]]
            + 0.4 * values["rlvr_0360"][split["test"]]
        )
        seed_results.append(
            {
                "seed": seed,
                "split": {
                    name: {
                        "prompts": int(len(indices)),
                        "clusters": int(len(np.unique(cluster_ids[indices]))),
                        "categories": {
                            str(category): int(np.sum(categories[indices] == category))
                            for category in sorted(np.unique(categories).tolist())
                        },
                    }
                    for name, indices in split.items()
                },
                "validation": reconstruction_metrics(
                    validation, validation_prediction, mean
                ),
                "test": reconstruction_metrics(test, test_prediction, mean),
                "test_by_category": category_results,
                "raw_chord_test": reconstruction_metrics(
                    middle, raw_chord, middle_mean
                ),
            }
        )

    test_scores = np.asarray(
        [row["test"]["r2_about_train_mean"] for row in seed_results]
    )
    interpolation_scores = np.asarray(
        [row["raw_chord_test"]["r2_about_train_mean"] for row in seed_results]
    )
    category_summary = {}
    for category in sorted(np.unique(categories).tolist()):
        scores = np.asarray(
            [
                row["test_by_category"][str(category)]["r2_about_train_mean"]
                for row in seed_results
            ]
        )
        category_summary[str(category)] = {
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)),
            "minimum": float(scores.min()),
            "maximum": float(scores.max()),
        }

    baseline = seed_results[0]
    baseline_split = stratified_cluster_split(cluster_ids, categories, 1729)
    baseline_train = pooled(values, names, baseline_split["train"])
    baseline_test = pooled(values, names, baseline_split["test"])
    baseline_mean, baseline_basis = _fit_pca(
        baseline_train, args.rank, 1729 + 80_000 + args.layer
    )
    baseline_prediction = _project(baseline_test, baseline_mean, baseline_basis)
    row_clusters = np.concatenate([cluster_ids[baseline_split["test"]] for _ in names])
    bootstrap = cluster_bootstrap(
        baseline_test,
        baseline_prediction,
        baseline_mean,
        row_clusters,
        args.bootstrap_samples,
        91_729,
    )

    report = {
        "kind": "trajectory_split_and_category_robustness",
        "status": "exploratory_followup",
        "layer": args.layer,
        "rank": args.rank,
        "split_protocol": "category-stratified and cluster-disjoint 60/20/20",
        "seeds": args.seeds,
        "test_r2_across_seeds": {
            "mean": float(test_scores.mean()),
            "std": float(test_scores.std(ddof=1)),
            "minimum": float(test_scores.min()),
            "maximum": float(test_scores.max()),
            "fraction_at_least_0_70": float(np.mean(test_scores >= 0.70)),
        },
        "raw_chord_r2_across_seeds": {
            "mean": float(interpolation_scores.mean()),
            "std": float(interpolation_scores.std(ddof=1)),
            "minimum": float(interpolation_scores.min()),
            "maximum": float(interpolation_scores.max()),
        },
        "category_test_r2_across_seeds": category_summary,
        "seed_1729_cluster_bootstrap": bootstrap,
        "per_seed": seed_results,
        "runtime": runtime_manifest(),
    }
    output = ROOT / args.output
    write_json(report, output)
    print(json.dumps({key: value for key, value in report.items() if key != "per_seed"}, indent=2))
    print(f"Wrote robustness report: {output}")


if __name__ == "__main__":
    main()
