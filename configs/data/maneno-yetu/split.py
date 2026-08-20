#!/usr/bin/env python3
"""
Compute and persist a train/val/test split as a standalone manifest,
decoupled from the corpus files themselves.

- Reads filenames from --data-dir. Never copies, moves, or edits them.
- Writes only the manifest to --out-dir (must differ from --data-dir).
- Other scripts import load_split() to resolve the manifest back into
  real paths against whatever --data-dir they're given — laptop today,
  cluster later, same manifest either way.

Run from anywhere — defaults are anchored to this file's own location,
not the current working directory:
    python split.py
"""

import argparse
import json
import pathlib
import random
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent   # .../data/maneno-yetu/
DEFAULT_DATA_DIR = HERE / "data-raw" / "cleaned"
DEFAULT_OUT_DIR = HERE / "splits"


def discover_files(data_dir: pathlib.Path, pattern: str) -> list[str]:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")
    files = sorted(p.name for p in data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {data_dir}")
    return files


def compute_split(filenames, train_frac, val_frac, seed):
    assert 0 < train_frac < 1 and 0 <= val_frac < 1
    assert train_frac + val_frac < 1, "must leave room for a non-empty test set"

    rng = random.Random(seed)   # local instance — isolated from global random state
    shuffled = filenames[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_manifest(splits: dict, out_dir: pathlib.Path, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.json"
    if manifest_path.exists():
        print(f"WARNING: overwriting existing manifest at {manifest_path}")

    manifest_path.write_text(
        json.dumps({"meta": meta, "splits": splits}, indent=2, ensure_ascii=False))

    for name, files in splits.items():
        (out_dir / f"{name}_files.txt").write_text("\n".join(files) + "\n")

    return manifest_path


def load_split(manifest_path, data_dir: pathlib.Path):
    """What downstream scripts call. Returns {'train': [Path, ...], ...}."""
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    resolved, missing = {}, []
    for split_name, filenames in manifest["splits"].items():
        paths = [pathlib.Path(data_dir) / fn for fn in filenames]
        missing += [str(p) for p in paths if not p.exists()]
        resolved[split_name] = paths
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest entries missing under {data_dir}, e.g. {missing[0]}")
    return resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--pattern", default="maneno-yetu-corpora-*.txt")
    ap.add_argument("--train-frac", type=float, default=0.90)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out_dir.resolve() == args.data_dir.resolve():
        raise ValueError("--out-dir must differ from --data-dir — splits "
                          "are metadata, not data.")

    print(f"Reading from: {args.data_dir.resolve()}")
    print(f"Writing to:   {args.out_dir.resolve()}")

    filenames = discover_files(args.data_dir, args.pattern)
    splits = compute_split(filenames, args.train_frac, args.val_frac, args.seed)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_dir": str(args.data_dir.resolve()),
        "pattern": args.pattern,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": round(1 - args.train_frac - args.val_frac, 4),
        "n_total": len(filenames),
    }
    manifest_path = write_manifest(splits, args.out_dir, meta)

    print(f"\nWrote manifest to {manifest_path}")
    for name, files in splits.items():
        print(f"  {name}: {len(files)} files")


if __name__ == "__main__":
    main()