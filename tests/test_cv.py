"""CV purging/embargo correctness: no train index leaks into the test band."""

import pytest

from src import cv


@pytest.mark.parametrize("embargo", [0, 1, 2, 5])
@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("n_splits", [3, 5])
def test_walk_forward_no_leakage(embargo, horizon, n_splits):
    splits = cv.purged_walk_forward(
        200, n_splits=n_splits, embargo=embargo, horizon=horizon, min_train_size=20
    )
    assert len(splits) > 0
    for sp in splits:
        assert not cv.has_leakage(sp.train, sp.test, embargo=embargo, horizon=horizon)
        # Walk-forward: all training indices precede the test block.
        assert sp.train.max() < sp.test.min()
        # No overlap.
        assert len(set(sp.train).intersection(set(sp.test))) == 0


@pytest.mark.parametrize("embargo", [0, 1, 3])
@pytest.mark.parametrize("horizon", [1, 2])
def test_kfold_no_leakage(embargo, horizon):
    splits = cv.purged_kfold(150, n_splits=5, embargo=embargo, horizon=horizon)
    for sp in splits:
        assert not cv.has_leakage(sp.train, sp.test, embargo=embargo, horizon=horizon)
        assert len(set(sp.train).intersection(set(sp.test))) == 0


def test_embargo_creates_gap():
    splits = cv.purged_walk_forward(100, n_splits=4, embargo=3, horizon=2, min_train_size=10)
    for sp in splits:
        gap = sp.test.min() - sp.train.max()
        # Gap must exceed horizon + embargo.
        assert gap > 2 + 3
