from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "NousResearch/hermes-function-calling-v1"
DEFAULT_CONFIGS = ("func_calling_singleturn", "func_calling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hermes function-calling configs into local JSONL snapshots"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory for downloaded JSONL snapshots",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Hermes configs to download",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-config row limit for debugging",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    stats: dict[str, int] = {}
    for config in args.configs:
        dataset = load_dataset(DATASET_ID, config, split="train")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(dataset):
            if args.limit > 0 and index >= args.limit:
                break
            enriched_row = {
                "source_dataset": DATASET_ID,
                "source_config": config,
                "source_row_idx": index,
                **row,
            }
            rows.append(enriched_row)

        out_path = output_dir / f"hermes_{config}.jsonl"
        write_jsonl(out_path, rows)
        stats[config] = len(rows)
        print(f"wrote {len(rows)} rows -> {out_path}")

    stats_path = output_dir / "download_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote stats -> {stats_path}")


if __name__ == "__main__":
    main()
