import pytest

from alignment_manifold.trajectory import _interpolation_weight


def test_interpolation_weight_uses_training_steps() -> None:
    assert _interpolation_weight(200, 1400, 2600) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "steps",
    [
        (200, 200, 2600),
        (200, 2600, 1400),
        (2600, 1400, 200),
    ],
)
def test_interpolation_weight_requires_strict_order(steps: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _interpolation_weight(*steps)
