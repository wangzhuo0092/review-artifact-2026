"""Sampling for controlled perturbations and annotation budgets (Sec. 5.3–5.4)."""

from dataclasses import replace
import hashlib
import random

import numpy as np


PERTURBATIONS = ("random", "equal", "flip", "mixed")
PERTURBATION_RATIOS = tuple(p / 100 for p in range(0, 50, 5))
PERTURBATION_SEEDS = (42, 2023, 3407, 7919, 15401)
SCALE_RATER_BUDGETS = (2, 4, 8, 16, 32, 64, 112)
SCALE_COMPARISON_BUDGETS = (1, 2, 4, 8, 16, 32, 64)
SCALE_REPETITIONS = 100


def _dataset_seed_key(name):
    # Retain the source runner's dataset key for its deterministic draws.
    return "chatbot_arena_33k" if name == "chatbot_arena" else name


def _digest(*parts):
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()


def _rng(name, seed, channel):
    return np.random.default_rng(int(_digest(name, seed, channel)[:16], 16) % (2**32))


def _row_uniform(name, seed, kind, row, channel):
    return int(_digest(name, seed, kind, row, channel)[:16], 16) / 2**64


def _rater_prefix(counts, fraction, rng):
    """Choose a nested set of whole raters targeting a fraction of all labels."""
    targets = np.asarray(sorted({*PERTURBATION_RATIOS[1:], fraction}))
    best = None
    for _ in range(max(64, min(256, 4 * len(counts)))):
        order = rng.permutation(len(counts))
        coverage = np.cumsum(counts[order]) / counts.sum()
        coverage = coverage[coverage < 0.5]
        if not len(coverage):
            continue
        # The prefix length at each target is its nearest attainable coverage.
        # Searching sorted coverage only requires inspecting the adjacent entries.
        right = np.minimum(np.searchsorted(coverage, targets), len(coverage) - 1)
        left = np.maximum(right - 1, 0)
        prefix_indices = np.where(
            abs(coverage[left] - targets) <= abs(coverage[right] - targets), left, right
        )
        error = float(np.sum(abs(coverage[prefix_indices] - targets)))
        if best is None or error < best[0]:
            prefix = prefix_indices[np.searchsorted(targets, fraction)] + 1
            best = (error, order[:prefix])
    if best is None:
        raise ValueError("No whole-annotator selection covers fewer than half the labels")
    return best[1]


def perturbation_rows(data, fraction, seed, unit):
    """Return selected row indices before corruption, for all four label changes.

    The source experiments exclude calibration comparisons and same-item pairs.
    The rater protocol uses nested random prefixes chosen to approximate each
    requested label fraction while keeping every selected rater's rows together.
    """
    if unit not in ("comparison", "rater"):
        raise ValueError(f"Unknown perturbation unit: {unit}")
    if not 0 <= fraction < 0.5:
        raise ValueError("Perturbation fraction must be in [0, 0.5)")
    active_rows = np.flatnonzero((data.a != data.b) & ~_golden(data))
    if fraction == 0 or not len(active_rows):
        return np.empty(0, dtype=np.int64)
    name = _dataset_seed_key(data.name)
    if unit == "comparison":
        count = max(1, round(fraction * len(active_rows)))
        order = _rng(name, seed, "individual_label_order").permutation(len(active_rows))
        return np.sort(active_rows[order[:count]])

    annotator = data.annotator[active_rows]
    raters, counts = np.unique(annotator, return_counts=True)
    # Anonymous IDs preserve the original canonical rater order. Sorting also
    # keeps this selection independent of how the loader indexes first appearances.
    order = np.argsort([data.annotators[i] for i in raters], kind="stable")
    picked = _rater_prefix(counts[order], fraction, _rng(name, seed, "raters"))
    return active_rows[np.isin(annotator, raters[order[picked]])]


def _golden(data):
    values = getattr(data, "golden", None)
    return np.zeros(len(data.y), dtype=bool) if values is None else np.asarray(values, dtype=bool)


def perturb(data, kind, fraction, seed, unit):
    """Apply one Fig. 2/3 condition and return the data and exposed annotator IDs.

    An existing tie remains a tie under Random or Flip. Mixed independently
    chooses one of Random, Equal or Flip for each selected row. The exposed set
    includes selected raters even when some of their selected labels stay the same.
    """
    if kind not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {kind}")
    selected = perturbation_rows(data, fraction, seed, unit)
    y = data.y.copy()
    name = _dataset_seed_key(data.name)
    eligible = np.flatnonzero((data.a != data.b) & ~_golden(data))
    for index, row in zip(selected, np.searchsorted(eligible, selected)):
        change = kind
        if kind == "mixed":
            draw = _row_uniform(name, seed, kind, row, "mixed_kind")
            change = PERTURBATIONS[min(int(draw * 3), 2)]
        if change == "equal":
            y[index] = 0.5
        elif change == "flip":
            y[index] = 1.0 - y[index]
        elif y[index] != 0.5:
            draw = _row_uniform(name, seed, kind, row, "random_value")
            y[index] = 0.5 if draw < 0.5 else 1.0 - y[index]
    return replace(data, y=y), np.unique(data.annotator[selected])


def scale_seed(name, axis, budget, iteration, base_seed=42):
    """Use the original scale-dependent offsets for paired Fig. 4 trials."""
    key = _dataset_seed_key(name)
    return base_seed + int(_digest(key, axis, budget)[:8], 16) + iteration


def scale(data, axis, budget, seed):
    """Draw Fig. 4 annotators with replacement and clone each sampled record.

    Along the comparison axis, use the full-size annotator bootstrap and keep
    up to the requested number of ordinary comparisons per draw, sampled without
    replacement. Calibration rows stay attached and do not count toward this cap.
    """
    if axis not in ("n_raters", "comparisons_per_rater"):
        raise ValueError(f"Unknown scaling axis: {axis}")
    if budget < 1 or not len(data.y):
        raise ValueError("Scaling requires a positive budget and at least one row")
    clusters = {}
    for row, rater in enumerate(data.annotator):
        clusters.setdefault(int(rater), []).append(row)
    rng = random.Random(seed)
    size = budget if axis == "n_raters" else len(clusters)
    draws = rng.choices(list(clusters), k=size)
    golden = _golden(data)
    has_golden = np.any(golden)
    rows, clones = [], []
    for clone, rater in enumerate(draws):
        members = clusters[rater]
        if axis == "comparisons_per_rater" and len(members) > budget:
            ordinary = [row for row in members if not golden[row]]
            if len(ordinary) > budget:
                sampled = rng.sample(ordinary, budget)
                members = sorted(sampled + [row for row in members if golden[row]]) if has_golden else sampled
        rows.extend(members)
        clones.extend([clone] * len(members))
    return data.subset(np.asarray(rows, dtype=np.int64), clone=np.asarray(clones, dtype=np.int64))
