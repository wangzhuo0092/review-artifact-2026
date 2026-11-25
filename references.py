"""Load external reference leaderboards used by the resampling protocol."""

from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np


HUMAINE_COMMIT = "dff6be8201e15926a275fd953f44bbe8fb4aa499"
HUMAINE_URL = (
    "https://huggingface.co/spaces/ProlificAI/humaine-leaderboard/resolve/"
    f"{HUMAINE_COMMIT}/leaderboard-app/public/btd_frontend_data.json"
)
HUMAINE_SHA256 = "abec278dcb874cc5ae40298922b01f85b51aa2e0f51eaf149d55b0c6f98ec634"


def _align_scores(rows, items, source):
    """Validate exact model coverage and return scores in input item order."""
    items = list(items)
    if len(items) < 2 or len(set(items)) != len(items):
        raise ValueError("The experiment item list must contain at least two distinct model names")
    lookup = {}
    for name, score in rows:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Missing model name in {source}")
        if name in lookup:
            raise ValueError(f"Duplicate model name in {source}: {name!r}")
        try:
            value = float(score)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid score in {source} for {name!r}") from error
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite score in {source} for {name!r}")
        lookup[name] = value
    expected = set(items)
    missing = sorted(expected - lookup.keys())
    extra = sorted(lookup.keys() - expected)
    if missing or extra:
        raise ValueError(
            f"Model names in {source} must exactly match the experiment item list. "
            f"Missing: {missing}. Extra: {extra}."
        )
    return np.asarray([lookup[name] for name in items], dtype=float)


def reference_scores(path, items):
    """Load a reference CSV with `model_name` and `score` columns."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"model_name", "score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Reference CSV {path} must have model_name and score columns")
        return _align_scores(
            ((row["model_name"], row["score"]) for row in reader), items, str(path)
        )


@lru_cache(maxsize=1)
def _humaine_snapshot():
    try:
        with urlopen(HUMAINE_URL, timeout=30) as response:
            payload = response.read()
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            "Could not download the pinned HUMAINE reference. "
            "Supply a local reference CSV if the snapshot is unavailable."
        ) from error
    if hashlib.sha256(payload).hexdigest() != HUMAINE_SHA256:
        raise ValueError("The pinned HUMAINE reference failed SHA-256 verification")
    data = json.loads(payload)
    return tuple((row["model_name"], row["score"]) for row in data["overall_rankings"])


def humaine_scores(items):
    """Download and align the pinned public HUMAINE overall leaderboard."""
    return _align_scores(_humaine_snapshot(), items, "the pinned HUMAINE snapshot")
