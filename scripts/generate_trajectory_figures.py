from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, Rectangle
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alignment_manifold.config import load_config  # noqa: E402
from alignment_manifold.geometry import _split_indices  # noqa: E402


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#5B6770"
LIGHT_GRAY = "#D7DCE0"
CHECKPOINT_COLORS = [BLUE, ORANGE, GREEN]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": stem.replace("_", " ").title(),
        "Author": "alignment-manifold experiment",
        "Subject": "OLMo 2 multi-checkpoint activation geometry",
    }
    fig.savefig(
        output_dir / f"{stem}.png",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "matplotlib; scripts/generate_trajectory_figures.py"},
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata=metadata,
    )
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def figure_layer_rank_heatmap(report: dict, output_dir: Path) -> None:
    layers = sorted({row["layer"] for row in report["layer_rank_results"]})
    ranks = sorted({row["rank"] for row in report["layer_rank_results"]})
    values = np.full((len(layers), len(ranks)), np.nan)
    for row in report["layer_rank_results"]:
        values[layers.index(row["layer"]), ranks.index(row["rank"])] = row["test"][
            "r2_about_train_mean"
        ]

    fig, ax = plt.subplots(figsize=(7.4, 7.2), constrained_layout=True)
    image = ax.imshow(values, cmap="viridis", norm=Normalize(0.0, 1.0), aspect="auto")
    ax.set_xticks(range(len(ranks)), labels=ranks)
    ax.set_yticks(range(len(layers)), labels=layers)
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("Transformer layer (zero-indexed)")
    ax.set_title("Held-out reconstruction across layers and ranks")
    for layer_index in range(len(layers)):
        for rank_index in range(len(ranks)):
            value = values[layer_index, rank_index]
            color = "white" if value < 0.48 else "black"
            ax.text(
                rank_index,
                layer_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    primary_y = layers.index(report["primary_layer"])
    selected_x = ranks.index(report["selected_rank"])
    ax.add_patch(
        Rectangle(
            (selected_x - 0.48, primary_y - 0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor=VERMILLION,
            linewidth=2.2,
        )
    )
    cbar = fig.colorbar(image, ax=ax, shrink=0.83, pad=0.03)
    cbar.set_label("Centered held-out $R^2$")
    ax.text(
        0.0,
        -0.09,
        (
            f"Outlined cell: preregistered layer {report['primary_layer']}, "
            f"validation-selected rank {report['selected_rank']}."
        ),
        transform=ax.transAxes,
        fontsize=8,
        color=GRAY,
    )
    save_figure(fig, "fig01_layer_rank_heatmap", output_dir)


def figure_primary_rank_curve(report: dict, output_dir: Path) -> None:
    rows = sorted(
        [row for row in report["layer_rank_results"] if row["layer"] == report["primary_layer"]],
        key=lambda row: row["rank"],
    )
    ranks = np.asarray([row["rank"] for row in rows])
    validation = np.asarray([row["validation"]["r2_about_train_mean"] for row in rows])
    test = np.asarray([row["test"]["r2_about_train_mean"] for row in rows])
    control = report["global_pca"]["random_test_r2"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(ranks, validation, marker="o", color=ORANGE, label="Validation")
    ax.plot(ranks, test, marker="s", color=BLUE, label="Held-out test")
    ax.axhline(0.70, color=GRAY, linestyle="--", linewidth=1.2, label="Decision threshold (0.70)")
    ax.errorbar(
        [report["selected_rank"]],
        [control["mean"]],
        yerr=[control["std"]],
        fmt="^",
        markersize=7,
        capsize=4,
        color=VERMILLION,
        label="Random rank-32 controls (mean ± SD)",
    )
    ax.scatter(
        [report["selected_rank"]],
        [control["maximum"]],
        marker="x",
        s=42,
        color=VERMILLION,
        label="Best random control",
        zorder=4,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, labels=ranks)
    ax.set_ylim(-0.05, 0.76)
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("Centered $R^2$")
    ax.set_title(f"Layer {report['primary_layer']} rank curve and random-subspace control")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(loc="upper left", ncol=2)
    ax.annotate(
        f"Selected rank {report['selected_rank']}\nTest $R^2$ = {test[-1]:.3f}",
        xy=(ranks[-1], test[-1]),
        xytext=(11, -35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1},
        fontsize=8.5,
    )
    save_figure(fig, "fig02_primary_layer_rank_curve", output_dir)


def figure_subspace_similarity(report: dict, output_dir: Path) -> None:
    values = np.asarray(report["checkpoint_local_pca"]["subspace_similarity_matrix"])
    names = [name.replace("rlvr_", "step ") for name in report["checkpoint_order"]]
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    off_diagonal = values[~np.eye(len(values), dtype=bool)]
    color_min = max(0.0, float(np.floor(off_diagonal.min() * 20.0) / 20.0))
    image = ax.imshow(values, cmap="magma", vmin=color_min, vmax=1.0)
    ax.set_xticks(range(len(names)), labels=names)
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_xlabel("Checkpoint-local rank-32 subspace")
    ax.set_ylabel("Checkpoint-local rank-32 subspace")
    ax.set_title("Pairwise projection similarity of local subspaces")
    for row in range(len(names)):
        for column in range(len(names)):
            ax.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if values[row, column] < 0.94 else "black",
                fontsize=10,
            )
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.04)
    cbar.set_label("Mean squared canonical overlap")
    save_figure(fig, "fig03_subspace_similarity", output_dir)


def figure_model_comparison(report: dict, output_dir: Path) -> None:
    names = ["Global PCA", "Local PCA", "Autoencoder"]
    records = [
        report["global_pca"]["test"],
        report["checkpoint_local_pca"]["test"],
        report["autoencoder"]["test"],
    ]
    r2 = [record["r2_about_train_mean"] for record in records]
    raw = [record["fraction_raw_energy_reconstructed"] for record in records]
    colors = [BLUE, ORANGE, GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3), constrained_layout=True)
    for ax, values, title, ylabel in [
        (axes[0], r2, "Centered reconstruction", "Held-out $R^2$"),
        (axes[1], raw, "Raw activation energy", "Fraction reconstructed"),
    ]:
        bars = ax.bar(names, values, color=colors, width=0.66, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0, 0.75 if ax is axes[0] else 0.70)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
        ax.tick_params(axis="x", rotation=15)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8.5)
    axes[0].axhline(0.70, color=GRAY, linestyle="--", linewidth=1.2)
    axes[0].text(2.45, 0.705, "0.70 threshold", ha="right", va="bottom", fontsize=8, color=GRAY)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle("Shared linear, checkpoint-local, and nonlinear reconstruction", fontsize=11.5)
    save_figure(fig, "fig04_model_comparison", output_dir)


def figure_interpolation(report: dict, config: dict, output_dir: Path) -> None:
    interpolation = report["interpolation"]
    keys = ["raw_activation_chord", "global_pca_chord", "autoencoder_latent_chord"]
    names = ["Raw chord", "PCA chord", "Latent chord"]
    colors = [BLUE, ORANGE, GREEN]
    r2 = [interpolation[key]["r2_about_train_mean"] for key in keys]
    error = [interpolation[key]["normalized_frobenius_error"] for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3), constrained_layout=True)
    bars = axes[0].bar(names, r2, color=colors, width=0.66, edgecolor="white")
    axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in r2], padding=3, fontsize=8.5)
    axes[0].set_ylim(0, 1.05)
    middle_step = report["training_steps"][config["trajectory"]["interpolation"]["middle"]]
    axes[0].set_ylabel(f"Step-{middle_step} centered $R^2$")
    axes[0].set_title("Midpoint prediction accuracy")
    axes[0].grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
    bars = axes[1].bar(names, error, color=colors, width=0.66, edgecolor="white")
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in error], padding=3, fontsize=8.5)
    axes[1].set_ylim(0, 0.82)
    axes[1].set_ylabel("Normalized Frobenius error (lower is better)")
    axes[1].set_title("Midpoint prediction error")
    axes[1].grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    interpolation_config = config["trajectory"]["interpolation"]
    steps = report["training_steps"]
    left_step = steps[interpolation_config["left"]]
    middle_step = steps[interpolation_config["middle"]]
    right_step = steps[interpolation_config["right"]]
    fig.suptitle(
        f"Interpolation of RLVR step {middle_step} from steps {left_step} and {right_step}",
        fontsize=11.5,
    )
    save_figure(fig, "fig05_interpolation_comparison", output_dir)


def confidence_ellipse_of_mean(
    values: np.ndarray, ax: plt.Axes, color: str, confidence_scale: float = np.sqrt(5.991)
) -> None:
    covariance = np.cov(values.T, ddof=1) / len(values)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * confidence_scale * np.sqrt(eigenvalues)
    center = values.mean(axis=0)
    ax.add_patch(
        Ellipse(
            center,
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.16,
            linewidth=1.2,
        )
    )


def figure_test_trajectory(
    report: dict, config: dict, extraction_dir: Path, output_dir: Path
) -> None:
    layer = int(report["primary_layer"])
    reference_path = extraction_dir / f"{report['reference_checkpoint']}.npz"
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = archive["activations"][:, layer].astype(np.float32)
        cluster_ids = archive["cluster_ids"]
    split = _split_indices(
        cluster_ids,
        float(config["data"]["train_fraction"]),
        float(config["data"]["validation_fraction"]),
        int(config["experiment"]["seed"]),
    )
    deltas: dict[str, np.ndarray] = {}
    for name in report["checkpoint_order"]:
        with np.load(extraction_dir / f"{name}.npz", allow_pickle=False) as archive:
            deltas[name] = archive["activations"][:, layer].astype(np.float32) - reference
    train = np.concatenate([deltas[name][split["train"]] for name in report["checkpoint_order"]])
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(train - mean)
    projected = {
        name: pca.transform(deltas[name][split["test"]] - mean)
        for name in report["checkpoint_order"]
    }

    fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)
    for prompt_index in range(len(split["test"])):
        trajectory = np.vstack([projected[name][prompt_index] for name in report["checkpoint_order"]])
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=GRAY, alpha=0.13, linewidth=0.7, zorder=1)
    for color, name in zip(CHECKPOINT_COLORS, report["checkpoint_order"]):
        values = projected[name]
        label = name.replace("rlvr_", "step ")
        ax.scatter(
            values[:, 0],
            values[:, 1],
            s=15,
            color=color,
            alpha=0.42,
            edgecolors="none",
            label=f"{label} test prompts",
            zorder=2,
        )
        confidence_ellipse_of_mean(values, ax, color)
        centroid = values.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=85,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            marker="D",
            zorder=4,
        )
    centroids = np.vstack([projected[name].mean(axis=0) for name in report["checkpoint_order"]])
    ax.plot(
        centroids[:, 0],
        centroids[:, 1],
        color="black",
        linestyle="--",
        linewidth=1.2,
        zorder=3,
        label="Checkpoint centroids",
    )
    ax.axhline(0, color=LIGHT_GRAY, linewidth=0.7, zorder=0)
    ax.axvline(0, color=LIGHT_GRAY, linewidth=0.7, zorder=0)
    ax.set_xlabel(f"Training-fitted PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)")
    ax.set_ylabel(f"Training-fitted PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)")
    ax.set_title("Held-out prompt trajectories in a training-fitted 2D projection")
    ax.legend(loc="best")
    ax.text(
        0.0,
        -0.12,
        "Thin lines connect the same held-out prompt; shaded ellipses are 95% confidence regions for checkpoint means.",
        transform=ax.transAxes,
        fontsize=8,
        color=GRAY,
    )
    save_figure(fig, "fig06_test_prompt_trajectory", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trajectory report figures")
    parser.add_argument("--config", default="configs/trajectory.yaml")
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()
    configure_style()
    config_path = (ROOT / args.config).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    config = load_config(config_path)
    artifact_root = (ROOT / config["experiment"]["artifact_dir"]).resolve()
    report_path = artifact_root / "trajectory_geometry" / "report.json"
    extraction_dir = artifact_root / "extractions"
    if not report_path.exists():
        raise FileNotFoundError(f"Run trajectory geometry first: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    figure_layer_rank_heatmap(report, output_dir)
    figure_primary_rank_curve(report, output_dir)
    figure_subspace_similarity(report, output_dir)
    figure_model_comparison(report, output_dir)
    figure_interpolation(report, config, output_dir)
    figure_test_trajectory(report, config, extraction_dir, output_dir)
    print(f"Wrote 6 PNG and 6 PDF figures to {output_dir}")


if __name__ == "__main__":
    main()
