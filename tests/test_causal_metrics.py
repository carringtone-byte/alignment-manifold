import numpy as np

from alignment_manifold.causal import (
    kl_divergence,
    normalized_gap_recovery,
    paired_bidirectional_bootstrap,
)


def test_normalized_gap_recovery_endpoints() -> None:
    target = np.asarray([[2.0, 0.0, -1.0], [0.0, 1.0, -2.0]])
    baseline = np.asarray([[0.0, 2.0, -1.0], [1.0, 0.0, -2.0]])
    unchanged = normalized_gap_recovery(target, baseline, baseline)
    perfect = normalized_gap_recovery(target, baseline, target)
    assert abs(unchanged["aggregate_recovery"]) < 1e-10
    assert abs(perfect["aggregate_recovery"] - 1.0) < 1e-10
    assert np.all(kl_divergence(target, target) < 1e-12)


def test_paired_bootstrap_preserves_perfect_recovery() -> None:
    target = np.asarray([[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    baseline = np.asarray([[0.0, 2.0], [2.0, 0.0], [-1.0, 1.0]])
    perfect = normalized_gap_recovery(target, baseline, target)
    bootstrap = paired_bidirectional_bootstrap(perfect, perfect, seed=2, samples=100)
    assert abs(bootstrap["mean"] - 1.0) < 1e-12
    assert abs(bootstrap["ci_025"] - 1.0) < 1e-12
    assert abs(bootstrap["ci_975"] - 1.0) < 1e-12
