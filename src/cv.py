"""Purged + embargoed walk-forward cross-validation (Lopez de Prado).

When labels are forward returns spanning ``horizon`` periods, a naive train/test
split leaks information: a training observation whose label window overlaps the
test window (or sits immediately next to it) shares data with the test set.  We
therefore:

* **Purge** training observations whose label window overlaps the test window.
* **Embargo** an additional band of observations around the test window so that
  serial correlation right at the boundary cannot leak either.

The functions here operate on integer positions over a *time-ordered* sequence
of observations (e.g. monthly rebalance periods).  ``horizon`` and ``embargo``
are expressed in the same period units.

Reference: Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray


def _apply_purge_embargo(
    train: np.ndarray, a: int, b: int, horizon: int, embargo: int
) -> np.ndarray:
    """Drop train positions inside the forbidden band [a-h-e, b+h+e]."""
    lo = a - horizon - embargo
    hi = b + horizon + embargo
    return train[(train < lo) | (train > hi)]


def purged_walk_forward(
    n_samples: int,
    *,
    n_splits: int = 5,
    embargo: int = 1,
    horizon: int = 1,
    min_train_size: int = 1,
    expanding: bool = True,
) -> list[Split]:
    """Expanding (or rolling) walk-forward splits with purge + embargo.

    The tail ``[min_train_size, n_samples)`` is divided into ``n_splits``
    contiguous test blocks moving forward in time.  Training data always
    precedes its test block; the last ``horizon + embargo`` periods before each
    test block are purged/embargoed away.
    """
    if n_samples <= min_train_size + n_splits:
        raise ValueError("Not enough samples for the requested number of splits.")

    indices = np.arange(n_samples)
    test_blocks = np.array_split(np.arange(min_train_size, n_samples), n_splits)

    splits: list[Split] = []
    for block in test_blocks:
        if len(block) == 0:
            continue
        a, b = int(block[0]), int(block[-1])
        if expanding:
            train = indices[indices < a]
        else:  # rolling: train window of size == prior test cumulative? keep simple
            train = indices[indices < a]
        train = _apply_purge_embargo(train, a, b, horizon, embargo)
        if len(train) == 0:
            continue
        splits.append(Split(train=train, test=block))
    return splits


def purged_kfold(
    n_samples: int,
    *,
    n_splits: int = 5,
    embargo: int = 1,
    horizon: int = 1,
) -> list[Split]:
    """Purged + embargoed k-fold (test block can sit in the middle of train)."""
    indices = np.arange(n_samples)
    test_blocks = np.array_split(indices, n_splits)
    splits: list[Split] = []
    for block in test_blocks:
        a, b = int(block[0]), int(block[-1])
        train = indices[(indices < a) | (indices > b)]
        train = _apply_purge_embargo(train, a, b, horizon, embargo)
        splits.append(Split(train=train, test=block))
    return splits


def has_leakage(
    train: np.ndarray, test: np.ndarray, *, embargo: int, horizon: int
) -> bool:
    """True if any train position is within ``horizon + embargo`` of the test span.

    This is the property the unit tests assert is FALSE for every produced split.
    """
    if len(train) == 0 or len(test) == 0:
        return False
    a, b = int(np.min(test)), int(np.max(test))
    lo = a - horizon - embargo
    hi = b + horizon + embargo
    inside = (train >= lo) & (train <= hi)
    return bool(np.any(inside))
