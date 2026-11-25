"""Load the anonymized comparison records used in the paper."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASETS = ("search_arena", "vision_arena", "chatbot_arena", "conha", "humaine",
            "ihq_all", "ihq_screened", "ihq_unscreened")
DATA_ROOT = Path(__file__).resolve().parent / "data"
OUTCOMES = {"model_a": 1.0, "model_b": 0.0, "tie": 0.5}


@dataclass(frozen=True)
class Data:
    name: str
    items: tuple
    annotators: tuple
    a: np.ndarray
    b: np.ndarray
    y: np.ndarray
    annotator: np.ndarray
    row: np.ndarray
    golden: np.ndarray
    correct: np.ndarray

    def subset(self, rows, *, clone=None):
        if clone is None:
            annotators, ids = self.annotators, self.annotator[rows]
        else:
            clone = np.asarray(clone, dtype=int)
            count = int(clone.max()) + 1 if len(clone) else 0
            # The original runner groups string clone identifiers lexicographically.
            ordered = sorted(range(count), key=lambda draw: f"user_{draw}")
            labels = np.empty(count, dtype=int)
            labels[ordered] = np.arange(count)
            annotators, ids = tuple(f"user_{draw}" for draw in ordered), labels[clone]
        return Data(self.name, self.items, annotators, self.a[rows], self.b[rows], self.y[rows],
                    ids, self.row[rows], self.golden[rows], self.correct[rows])


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_data(name, root=DATA_ROOT, *, calibration=False):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    root = Path(root)
    rows = _read(root / f"{name}.csv")
    if calibration and name.startswith("ihq"):
        stem = "ihq" if name == "ihq_all" else name
        golden = _read(root / f"{stem}_calibration.csv")
        by_source_index = {int(row["row"]): row for row in golden}
        count = len(rows) + len(golden)
        if len(by_source_index) != len(golden) or not all(0 <= i < count for i in by_source_index):
            raise ValueError(f"Invalid calibration row positions for {name}")
        regular_rows = iter(rows)
        rows = [
            by_source_index[i] if i in by_source_index else next(regular_rows)
            for i in range(count)
        ]
    items = tuple(sorted({row[side] for row in rows for side in ("model_a", "model_b")}))
    annotators = tuple(sorted({row["judge"] for row in rows}))
    item_ids = {item: index for index, item in enumerate(items)}
    user_ids = {user: index for index, user in enumerate(annotators)}
    return Data(
        name, items, annotators,
        np.array([item_ids[row["model_a"]] for row in rows], dtype=int),
        np.array([item_ids[row["model_b"]] for row in rows], dtype=int),
        np.array([OUTCOMES[row["winner"]] for row in rows]),
        np.array([user_ids[row["judge"]] for row in rows], dtype=int),
        np.arange(len(rows), dtype=int),
        np.array(["correct" in row for row in rows]),
        np.array([int(row.get("correct", -1)) for row in rows], dtype=int),
    )
