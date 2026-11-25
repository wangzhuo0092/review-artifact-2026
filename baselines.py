"""Adapters for the six ranking baselines in Section 5.1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from _baselines.am_elo import update_am_elo, update_label_smoothed_bt, update_m_elo
from _baselines.bayesian_elo import update_bayesian_elo as update_bayes_bt
from _baselines.bayesian_elo_noise_vectorized import update_bayesian_elo as update_bbq
from _baselines.models import Metric, QuerySet, Rating, Session, Slate, Stimulus

METHODS = {
    "crowd_bt": "Crowd-BT",
    "bayes_bt": "Bayes-BT",
    "bbq": "BBQ",
    "label_smoothing": "Label Smoothing",
    "m_elo": "m-ELO",
    "am_elo": "am-ELO",
}
LABEL_SMOOTHING = 0.10
_CROWD_DIR = Path(__file__).resolve().parent / "_baselines" / "crowd_bt"
_SCORE_LINE = re.compile(r"^m(\d+):\s*(\S+)\s*\[")
_RATER_LINE = re.compile(r"^u(\d+):\s*(\S+)\s*$")


@dataclass(frozen=True)
class BaselineFit:
    """Fitted scores in item ID order and quality in annotator ID order.

    A NaN marks an item or annotator missing from the fitting data.
    `converged` is None for the Crowd-BT binary, whose output lacks a
    documented convergence flag. Annotator quality uses the estimator's
    own parameterization, recorded in `quality_parameter`.
    `crowd_degenerate` records the diagnostic used by the controlled
    perturbation protocol.
    """

    scores: np.ndarray
    converged: bool | None
    reliability: np.ndarray | None = None
    quality_parameter: str | None = None
    crowd_degenerate: bool = False


def _crowd_bt(
    a, b, y, annotator, n_items, n_annotators, golden, crowd_settings,
):
    binary = Path(os.environ.get("REPAIR_CROWD_BT_BINARY", _CROWD_DIR / "build" / "elo_main"))
    if not binary.is_file():
        raise FileNotFoundError(
            "Build the Crowd-BT executable using _baselines/crowd_bt/README.md, "
            "or set REPAIR_CROWD_BT_BINARY to its location."
        )
    with TemporaryDirectory(prefix="repair-crowd-bt-") as directory:
        path = Path(directory) / "comparisons.csv"
        # Zero padding keeps numerically ordered item keys lexicographically sorted.
        width = max(1, len(str(n_items - 1)))
        with path.open("w", encoding="utf-8", newline="") as output:
            rows = csv.writer(output, lineterminator="\n")
            rows.writerow(
                ["methodA", "methodB", "isGolden", "modelProbability", "answerValue", "answerer"]
            )
            label = {0.0: "B", 0.5: "draw", 1.0: "A"}
            for i, j, outcome, rater in zip(a, b, y, annotator):
                rows.writerow(
                    [f"m{i:0{width}d}", f"m{j:0{width}d}", 0, 0, label[outcome], f"u{rater}"]
                )
            if golden is not None:
                gold_annotators, correct = golden
                gold_annotators = np.asarray(gold_annotators)
                correct = np.asarray(correct)
                if (
                    correct.ndim != 1 or correct.shape != gold_annotators.shape
                    or not np.issubdtype(gold_annotators.dtype, np.integer)
                    or np.any((gold_annotators < 0) | (gold_annotators >= n_annotators))
                    or not np.isin(correct, [0, 1]).all()
                ):
                    raise ValueError("Golden calibration must contain valid annotator IDs and 0/1 correctness")
                for rater, agrees in zip(gold_annotators, correct):
                    rows.writerow(["N/A", "N/A", 1, int(agrees), "draw", f"u{rater}"])
        command = [str(binary.resolve()), f"--csv_path={path}"]
        if crowd_settings:
            unknown = set(crowd_settings) - {"alpha", "rater_prior_strength"}
            if unknown:
                raise ValueError(f"Unknown Crowd-BT settings: {sorted(unknown)}")
            for name, value in sorted(crowd_settings.items()):
                value = float(value)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"Crowd-BT {name} must be finite and positive")
                command.append(f"--{name}={value}")
        process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode:
        raise RuntimeError(f"Crowd-BT exited with status {process.returncode}: {process.stderr.strip()}")
    scores = np.full(n_items, np.nan)
    reliability = np.full(n_annotators, np.nan)
    for line in process.stdout.splitlines():
        line = line.strip()
        if line == "Suggestions":
            break
        item = _SCORE_LINE.match(line)
        rater = _RATER_LINE.fullmatch(line)
        if item is not None:
            scores[int(item[1])] = float(item[2])
        elif rater is not None:
            reliability[int(rater[1])] = 1.0 - float(rater[2])
    _check_coverage(a, b, scores)
    observed_quality = reliability[np.isfinite(reliability)]
    finite_scores = scores[np.isfinite(scores)]
    degenerate = bool(
        observed_quality.size
        and np.mean(1.0 - observed_quality) >= 0.80
        and np.ptp(finite_scores) > 5000.0
    )
    return BaselineFit(
        scores, None, reliability, "nonrandom_probability", crowd_degenerate=degenerate,
    )


def _check_coverage(a, b, scores):
    present_items = np.unique(np.concatenate((a, b)))
    if not np.isfinite(scores[present_items]).all():
        raise RuntimeError("The estimator failed to return finite scores for all observed items")


def _sessions(a, b, y, annotator, ordered_annotators):
    """Group comparisons by annotator, retaining row order within sessions."""
    grouped = {int(rater): [] for rater in ordered_annotators}
    stimuli = {}
    for i, j, outcome, rater in zip(a, b, y, annotator):
        if int(i) not in stimuli:
            stimuli[int(i)] = Stimulus(str(i))
        if int(j) not in stimuli:
            stimuli[int(j)] = Stimulus(str(j))
        grouped[int(rater)].append(
            Slate([Rating(outcome, stimuli[int(i)]), Rating(1 - outcome, stimuli[int(j)])])
        )
    sessions = []
    for rater, slates in grouped.items():
        if slates:
            session = Session(slates, rater=str(rater))
            session.id = str(rater)
            sessions.append(session)
    return QuerySet(sessions)


def fit_baseline(
    method, a, b, y, annotator, n_items, *, seed=0, n_annotators=None,
    experiment="bootstrap", device="cpu", golden=None, crowd_settings=None,
    annotator_order=None,
):
    """Fit a baseline to 0-based item/annotator IDs and 0, 0.5, 1 outcomes.

    `seed` is accepted alongside the RePair runner's fitting arguments. These
    baselines use their own fixed initialization. The experiment argument is
    reserved for calling code. Pass Crowd-BT golden calibration explicitly as
    two arrays of annotator ID and correctness. `annotator_order` can specify
    the session processing order for the Python baselines.
    """
    if method == "ls_bt":
        method = "label_smoothing"
    if method not in METHODS:
        raise ValueError(f"Unknown baseline {method!r}")
    a, b, annotator = [np.asarray(value) for value in (a, b, annotator)]
    y = np.asarray(y, dtype=float)
    if any(value.ndim != 1 or value.shape != y.shape for value in (a, b, annotator)):
        raise ValueError("Items, annotator IDs, and outcomes must be equally sized 1-D arrays")
    if not y.size or not np.isin(y, [0.0, 0.5, 1.0]).all():
        raise ValueError("Provide one or more outcomes in {0, 0.5, 1}")
    if any(not np.issubdtype(value.dtype, np.integer) for value in (a, b, annotator)):
        raise ValueError("Item and annotator IDs must be integers")
    if np.any((a < 0) | (a >= n_items) | (b < 0) | (b >= n_items)):
        raise ValueError("Item ID outside the provided item range")
    if n_annotators is None:
        n_annotators = int(annotator.max()) + 1
    if np.any((annotator < 0) | (annotator >= n_annotators)):
        raise ValueError("Annotator ID outside the provided annotator range")
    if method == "crowd_bt":
        return _crowd_bt(
            a, b, y, annotator, n_items, n_annotators, golden, crowd_settings,
        )

    observed_annotators = np.unique(annotator)
    if annotator_order is None:
        annotator_order = observed_annotators
    else:
        annotator_order = np.asarray(annotator_order)
        if (
            annotator_order.ndim != 1
            or not np.issubdtype(annotator_order.dtype, np.integer)
            or len(annotator_order) != len(np.unique(annotator_order))
            or np.any((annotator_order < 0) | (annotator_order >= n_annotators))
            or not np.isin(observed_annotators, annotator_order).all()
        ):
            raise ValueError("annotator_order must list every observed annotator ID exactly once")
    sessions = _sessions(a, b, y, annotator, annotator_order)
    metric = Metric()
    optimizer_key = None
    if method == "bayes_bt":
        update_bayes_bt(metric, sessions, device=device)
    elif method == "bbq":
        update_bbq(metric, sessions, device=device)
    elif method == "label_smoothing":
        update_label_smoothed_bt(metric, sessions, label_smoothing=LABEL_SMOOTHING)
        optimizer_key = "ls_bt_optimization"
    elif method == "m_elo":
        update_m_elo(metric, sessions)
        optimizer_key = "m_elo_optimization"
    elif method == "am_elo":
        update_am_elo(metric, sessions)
        optimizer_key = "am_elo_optimization"

    state = metric.state
    converged = state.get("converged")
    if optimizer_key is not None:
        converged = state.get(optimizer_key, {}).get("success")
    scores = np.full(n_items, np.nan)
    for item, fit in state["scores"].items():
        scores[int(item)] = fit["value"]
    _check_coverage(a, b, scores)
    reliability = None
    if state["qualities"]:
        reliability = np.full(n_annotators, np.nan)
        for rater, fit in state["qualities"].items():
            reliability[int(rater)] = fit["value"]
    quality_parameter = state.get("quality_parameter")
    if method == "bbq" and quality_parameter is None:
        quality_parameter = "nonrandom_probability"
    return BaselineFit(
        scores, None if converged is None else bool(converged), reliability, quality_parameter,
    )
