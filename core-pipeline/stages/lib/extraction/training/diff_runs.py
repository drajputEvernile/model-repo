"""Selected value per page and field in two runs, side by side (from their candidate logs).

Usage:
  python -m stages.lib.extraction.training.diff_runs KV_Run_A KV_Run_B
Run names are folders under config.Run_Output (or full paths).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import run_selected
from ..util import config

PAGE = ["record_id", "page_number", "field"]


def _run_dir(name: str) -> Path:
    path = Path(name)
    return path if path.is_dir() else config.Run_Output / name


def selected(run: Path) -> pd.DataFrame:
    """One row per (page, field): the selected value(s), '' when nothing was selected."""
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in (run / "candidates").glob("*.csv")]
    if not frames:
        raise SystemExit(f"no candidate log under {run}")
    log = pd.concat(frames, ignore_index=True)
    picked = log[log.apply(run_selected, axis=1)] if len(log) else log
    values = (
        picked.groupby(PAGE)
        .agg(value=("value", lambda s: " | ".join(sorted(set(s)))),
             norm=("value_norm", lambda s: " | ".join(sorted(set(s)))),
             key=("key", lambda s: " | ".join(sorted(set(s)))))
        .reset_index()
    )
    pages = log[PAGE].drop_duplicates()
    return pages.merge(values, on=PAGE, how="left").fillna("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff selected values between two runs.")
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    args = parser.parse_args()
    a, b = _run_dir(args.run_a), _run_dir(args.run_b)
    both = selected(a).merge(selected(b), on=PAGE, how="outer", suffixes=("_a", "_b")).fillna("")
    changed = both[both["norm_a"] != both["norm_b"]]

    print(f"A = {a.name}\nB = {b.name}\n")
    print(f"{'field':22s} {'pages':>5s} {'same':>5s} {'changed':>7s} {'only A':>6s} {'only B':>6s}")
    for field, group in both.groupby("field"):
        diff = group[group["norm_a"] != group["norm_b"]]
        print(
            f"{field:22s} {len(group):5d} {len(group) - len(diff):5d} "
            f"{int(((diff['norm_a'] != '') & (diff['norm_b'] != '')).sum()):7d} "
            f"{int((diff['norm_b'] == '').sum()):6d} {int((diff['norm_a'] == '').sum()):6d}"
        )
    if len(changed):
        print("\nchanged pages:")
        columns = ["record_id", "page_number", "field", "key_a", "value_a", "key_b", "value_b"]
        print(changed.sort_values(PAGE)[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
