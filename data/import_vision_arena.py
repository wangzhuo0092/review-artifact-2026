"""Create the exact experiment input from authorized VisionArena Parquet files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path


COLUMNS = ("model_a", "model_b", "winner", "judge")
WINNERS = {"model_a": "model_a", "model_b": "model_b", "tie": "tie", "tie (bothbad)": "tie"}
SNAPSHOT = Path(__file__).with_name("SHA256SUMS")


def convert(shards: Path) -> bytes:
    import pyarrow.parquet as parquet

    rows = []
    for i in range(31):
        path = shards / f"train-{i:05d}-of-00031.parquet"
        table = parquet.read_table(path, columns=list(COLUMNS))
        for row in table.to_pylist():
            model_a, model_b = row["model_a"], row["model_b"]
            if model_a is None or model_b is None:
                raise ValueError(f"Missing model name in {path.name}")
            winner = str(row["winner"]).strip().lower()
            if winner not in WINNERS:
                raise ValueError(f"Unknown outcome {winner!r} in {path.name}")
            judge = "" if row["judge"] is None else str(row["judge"])
            rows.append((str(model_a), str(model_b), WINNERS[winner], judge if judge.strip() else "global_rater"))

    aliases = {
        judge: f"judge_{i:05d}" for i, judge in enumerate(sorted({row[3] for row in rows}))
    }
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows((a, b, winner, aliases[judge]) for a, b, winner, judge in rows)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_dir", type=Path, help="Directory with the 31 authorized Parquet shards")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("vision_arena.csv"))
    args = parser.parse_args()

    try:
        expected = next(
            line.split()[0]
            for line in SNAPSHOT.read_text(encoding="utf-8").splitlines()
            if line.split()[1] == "vision_arena.csv"
        )
        payload = convert(args.parquet_dir)
    except (ImportError, OSError, ValueError, StopIteration) as error:
        parser.exit(1, f"{error}\n")
    if hashlib.sha256(payload).hexdigest() != expected:
        parser.exit(1, "The converted file differs from the experiment snapshot. Check the source release.\n")
    args.output.write_bytes(payload)


if __name__ == "__main__":
    main()
