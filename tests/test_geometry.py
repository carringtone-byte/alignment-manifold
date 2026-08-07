import numpy as np

from alignment_manifold.geometry import (
    _fit_pca,
    _project,
    _split_indices,
    reconstruction_r2,
)


def test_group_split_has_no_cluster_leakage() -> None:
    groups = np.repeat(np.asarray([f"g{i}" for i in range(50)]), 4)
    split = _split_indices(groups, train_fraction=0.6, validation_fraction=0.2, seed=7)
    assert {name: len(indices) for name, indices in split.items()} == {
        "train": 120,
        "validation": 40,
        "test": 40,
    }
    group_sets = {name: set(groups[indices]) for name, indices in split.items()}
    assert not group_sets["train"] & group_sets["validation"]
    assert not group_sets["train"] & group_sets["test"]
    assert not group_sets["validation"] & group_sets["test"]


def test_pca_recovers_planted_low_rank_shift() -> None:
    rng = np.random.default_rng(11)
    width = 64
    rank = 4
    basis, _ = np.linalg.qr(rng.standard_normal((width, rank)))
    mean = rng.standard_normal(width)
    train = mean + rng.standard_normal((160, rank)) @ basis.T
    test = mean + rng.standard_normal((40, rank)) @ basis.T
    fitted_mean, fitted_basis = _fit_pca(train.astype(np.float32), rank, seed=3)
    prediction = _project(test.astype(np.float32), fitted_mean, fitted_basis)
    assert reconstruction_r2(test, prediction, fitted_mean) > 0.999

