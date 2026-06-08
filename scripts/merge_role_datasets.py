"""
Merge multiple role-labeled CSVs into one training dataset.

Expected schema for each input CSV:
  - text
  - role

Default inputs:
  - data/job_roles_augmented.csv
  - data/job_roles_tech_from_resume.csv

Default output:
  - data/job_roles_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

def _safe_print(s: str) -> None:
    try:
        print(s)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((s + "\n").encode("utf-8", errors="replace"))


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        headers = {h.strip() for h in reader.fieldnames}
        if not {"text", "role"}.issubset(headers):
            raise ValueError(f"CSV missing required columns text/role: {path}")
        for row in reader:
            text = (row.get("text") or "").strip()
            role = (row.get("role") or "").strip()
            if not text or not role:
                continue
            yield text, role


def merge(inputs: list[Path], output: Path) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    kept = 0
    dropped = 0

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "role"])
        writer.writeheader()
        for p in inputs:
            if not p.exists():
                continue
            for text, role in _iter_rows(p):
                key = (text, role)
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
                writer.writerow({"text": text, "role": role})
                kept += 1
    return kept, dropped


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="*",
        default=[
            str(root / "data" / "job_roles_augmented.csv"),
            str(root / "data" / "job_roles_tech_from_resume.csv"),
        ],
        help="Input CSVs with columns text,role",
    )
    p.add_argument(
        "--out",
        default=str(root / "data" / "job_roles_combined.csv"),
        help="Output merged CSV",
    )
    args = p.parse_args()

    inputs = [Path(x) for x in args.inputs]
    out = Path(args.out)
    kept, dropped = merge(inputs, out)
    _safe_print(f"Output: {out}")
    _safe_print(f"Kept: {kept}")
    _safe_print(f"Dropped duplicates: {dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

