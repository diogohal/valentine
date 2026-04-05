"""Common metric implementations for Valentine.

Custom metrics can be created by subclassing :class:`Metric`. We use ``@dataclass``
with ``frozen=True`` so instances are hashable and comparable without boilerplate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .base_metric import Metric
from .metric_helpers import get_fp, get_tp_fn

# Public exports
__all__ = [
    "F1Score",
    "Precision",
    "PrecisionTopNPercent",
    "Recall",
    "RecallAtSizeofGroundTruth",
    "PersistentAccuracy",
    "MissingAccuracy",
    "NewAccuracy",
]

# Ground truth can be either (source_col, target_col) name pairs or
# full ColumnPair instances (table-aware comparison).
GroundTruth = Sequence[tuple[str, str]] | Sequence[tuple]


def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, guarding against division by zero."""
    return numerator / denominator if denominator else 0.0


@dataclass(eq=True, frozen=True)
class Precision(Metric):
    """Precision = TP / (TP + FP).

    Attributes
    ----------
    one_to_one : bool
        Whether to apply the one-to-one filter to the MatcherResults first.
    """

    one_to_one: bool = True

    def apply(self, matches: Any, ground_truth: GroundTruth) -> dict[str, float]:
        if self.one_to_one:
            matches = matches.one_to_one()

        tp, _ = get_tp_fn(matches, ground_truth)
        fp = get_fp(matches, ground_truth)
        precision = _safe_div(tp, tp + fp)
        return self.return_format(precision)


@dataclass(eq=True, frozen=True)
class Recall(Metric):
    """Recall = TP / (TP + FN).

    Attributes
    ----------
    one_to_one : bool
        Whether to apply the one-to-one filter to the MatcherResults first.
    """

    one_to_one: bool = True

    def apply(self, matches: Any, ground_truth: GroundTruth) -> dict[str, float]:
        if self.one_to_one:
            matches = matches.one_to_one()

        tp, fn = get_tp_fn(matches, ground_truth)
        recall = _safe_div(tp, tp + fn)
        return self.return_format(recall)


@dataclass(eq=True, frozen=True)
class F1Score(Metric):
    """F1 score = 2 * (Precision * Recall) / (Precision + Recall).

    Attributes
    ----------
    one_to_one : bool
        Whether to apply the one-to-one filter to the MatcherResults first.
    """

    one_to_one: bool = True

    def apply(self, matches: Any, ground_truth: GroundTruth) -> dict[str, float]:
        if self.one_to_one:
            matches = matches.one_to_one()

        tp, fn = get_tp_fn(matches, ground_truth)
        fp = get_fp(matches, ground_truth)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * (precision * recall), (precision + recall))
        return self.return_format(f1)


@dataclass(eq=True, frozen=True)
class PrecisionTopNPercent(Metric):
    """Precision restricted to the top-N% of predicted matches by score.

    Attributes
    ----------
    one_to_one : bool
        Whether to apply the one-to-one filter to the MatcherResults first.
    n : int
        Percentage of matches to consider (0-100).
    """

    one_to_one: bool = True
    n: int = 10

    def name(self) -> str:
        # Replace the 'N' in the base name with the chosen percent, e.g. "PrecisionTop70Percent".
        return super().name().replace("N", str(self.n))

    def apply(self, matches: Any, ground_truth: GroundTruth) -> dict[str, float]:
        if self.one_to_one:
            matches = matches.one_to_one()

        # Clamp N to a sensible range without mutating the dataclass.
        n_clamped = min(100, max(0, int(self.n)))
        n_matches = matches.take_top_percent(n_clamped)

        tp, _ = get_tp_fn(n_matches, ground_truth)
        fp = get_fp(n_matches, ground_truth)
        precision_top_n_percent = _safe_div(tp, tp + fp)
        return self.return_format(precision_top_n_percent)


@dataclass(eq=True, frozen=True)
class RecallAtSizeofGroundTruth(Metric):
    """Recall when considering the top-|GT| predictions.

    This simulates selecting as many predictions as there are gold pairs and
    computing recall at that cutoff: TP / (TP + FN) where the candidate set is
    the top-``len(ground_truth)`` matches by score.

    Attributes
    ----------
    one_to_one : bool
        Whether to apply the one-to-one filter to the MatcherResults first.
    """

    one_to_one: bool = False

    def apply(self, matches: Any, ground_truth: GroundTruth) -> dict[str, float]:
        if self.one_to_one:
            matches = matches.one_to_one()
        n_matches = matches.take_top_n(len(ground_truth))
        tp, fn = get_tp_fn(n_matches, ground_truth)
        recall = _safe_div(tp, tp + fn)
        return self.return_format(recall)


@dataclass(eq=True, frozen=True)
class PersistentAccuracy(Metric):
    """Accuracy over persistent columns — columns whose name appears on both
    sides of the ground truth (i.e. self-matches such as ("A", "A")).

    For each such column the metric checks whether at least one predicted match
    maps that column to itself (same column name on both the source and target
    side).  The score is ``correct / total``, where *total* is the number of
    persistent columns derived from the ground truth.

    Returns ``-1.0`` when no persistent columns are found in the ground truth.
    """

    def apply(self, matches: Any, ground_truth: GroundTruth) -> dict[str, float]:
        persistent_cols = {src for src, tgt in ground_truth if src == tgt}

        total = len(persistent_cols)
        if total == 0:
            return self.return_format(-1.0)

        # Build a set of (source_col_name, target_col_name) pairs present in matches.
        predicted_pairs = {(key.source_column, key.target_column) for key in matches}

        correct = sum(
            1 for col in persistent_cols if (col, col) in predicted_pairs
        )
        return self.return_format(_safe_div(correct, total))


@dataclass(eq=True, frozen=True)
class MissingAccuracy(Metric):
    """Accuracy over missing columns — source columns that have no counterpart
    in the target schema (i.e. columns absent from the ground truth source side).

    A missing column is considered *correctly identified* when it does not appear
    as a source column in any predicted match.  The score is
    ``correctly_absent / total_missing``.

    Returns ``-1.0`` when no missing columns are found.

    Attributes
    ----------
    source_columns : tuple[str, ...]
        All column names present in the source table.
    """

    source_columns: tuple[str, ...] = ()

    def apply(self, matches: Any, ground_truth: GroundTruth) -> dict[str, float]:
        ground_truth_sources = {src for src, _ in ground_truth}
        missing_cols = set(self.source_columns) - ground_truth_sources

        total = len(missing_cols)
        if total == 0:
            return self.return_format(-1.0)

        # Source column names that appear in at least one predicted match.
        predicted_sources = {key.source_column for key in matches}

        correctly_absent = sum(
            1 for col in missing_cols if col not in predicted_sources
        )
        return self.return_format(_safe_div(correctly_absent, total))


@dataclass(eq=True, frozen=True)
class NewAccuracy(Metric):
    """Accuracy over new columns — target columns that have no counterpart in
    the source schema (i.e. columns absent from the ground truth target side).

    A new column is considered *correctly identified* when it does not appear
    as a target column in any predicted match.  The score is
    ``correctly_absent / total_new``.

    Returns ``-1.0`` when no new columns are found.

    Attributes
    ----------
    target_columns : tuple[str, ...]
        All column names present in the target table.
    """

    target_columns: tuple[str, ...] = ()

    def apply(self, matches: Any, ground_truth: GroundTruth) -> dict[str, float]:
        ground_truth_targets = {tgt for _, tgt in ground_truth}
        new_cols = set(self.target_columns) - ground_truth_targets

        total = len(new_cols)
        if total == 0:
            return self.return_format(-1.0)

        # Target column names that appear in at least one predicted match.
        predicted_targets = {key.target_column for key in matches}

        correctly_absent = sum(
            1 for col in new_cols if col not in predicted_targets
        )
        return self.return_format(_safe_div(correctly_absent, total))


