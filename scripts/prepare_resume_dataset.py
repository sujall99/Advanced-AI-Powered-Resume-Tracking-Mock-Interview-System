"""
Prepare a labeled resume dataset for the JobRoleClassifier.

Input (raw):
  - data new/raw/Resume.csv
  - data new/raw/Resume_extended.csv

Output (clean):
  - data/job_roles_from_resume.csv with columns: text, role (category labels)
  - data/job_roles_tech_from_resume.csv with columns: text, role (mapped tech roles)

Cleaning steps:
  - Prefer Resume_extended.csv if present
  - Drop rows with missing/empty text or role
  - Drop very short resumes (default < 200 chars)
  - Deduplicate by ID (if available) and by exact resume text
  - Normalize role labels (trim + collapse whitespace)
  - Optional: map broad categories into a smaller set of tech roles
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _safe_print(s: str) -> None:
    """
    Windows PowerShell consoles can be configured with a legacy codepage.
    Avoid crashing when printing paths containing non-ASCII characters.
    """
    try:
        print(s)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((s + "\n").encode("utf-8", errors="replace"))


def _pick_input(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data new" / "raw" / "Resume_extended.csv",
        repo_root / "data new" / "raw" / "Resume.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No raw resume CSV found. Expected one of:\n"
        f"- {candidates[0]}\n"
        f"- {candidates[1]}"
    )


def _iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    # Allow very long fields (some resumes are huge).
    try:
        csv.field_size_limit(1024 * 1024 * 50)  # 50MB
    except Exception:
        pass

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        for row in reader:
            yield {k: (v if v is not None else "") for k, v in row.items()}

def _map_to_tech_role(category: str, text: str) -> Optional[str]:
    """
    Heuristic mapping from broad dataset categories to the tech roles used by the app.
    Returns None when we can't confidently map (row will be dropped in tech mode).
    """
    cat = _norm_space(category).upper()
    t = (text or "").lower()

    # Only map categories that are reasonably "tech" in this dataset.
    if cat not in {"INFORMATION-TECHNOLOGY", "ENGINEERING"}:
        return None

    # Order matters: pick specific roles before generic SWE.
    if re.search(r"\b(airflow|spark|hadoop|kafka|etl|data lake|datalake|warehouse|redshift|bigquery|snowflake)\b", t):
        return "Data Engineer"
    if re.search(r"\b(machine learning|deep learning|nlp|computer vision|tensorflow|pytorch|scikit-learn|xgboost)\b", t):
        return "Data Scientist"
    if re.search(r"\b(tableau|power bi|dashboard|data analysis|reporting|excel|business intelligence|bi)\b", t):
        return "Data Analyst"
    if re.search(r"\b(kubernetes|docker|terraform|ansible|jenkins|ci/cd|prometheus|grafana|sre)\b", t):
        return "DevOps Engineer"
    if re.search(r"\b(react|next\\.js|vue|angular|typescript|frontend|html|css)\b", t):
        return "Frontend Developer"
    if re.search(r"\b(django|flask|fastapi|spring boot|express|node\\.js|rest api|microservices|backend)\b", t):
        return "Backend Developer"
    if re.search(r"\b(full[- ]stack|mern|mean)\b", t):
        return "Full Stack Developer"

    return "Software Engineer"


def prepare(
    in_path: Path,
    out_path: Path,
    *,
    min_chars: int = 200,
    text_col: str = "Resume_str",
    role_col: str = "Category",
    id_col: str = "ID",
    role_mode: str = "category",  # category | tech
) -> Tuple[int, int]:
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["text", "role"])
        writer.writeheader()

        for row in _iter_rows(in_path):
            raw_text = (row.get(text_col) or "").strip()
            raw_role = (row.get(role_col) or "").strip()
            raw_id = (row.get(id_col) or "").strip()

            if not raw_text or not raw_role:
                dropped += 1
                continue

            text = raw_text.strip()
            role = _norm_space(raw_role)

            if len(text) < int(min_chars):
                dropped += 1
                continue

            # ID de-dupe (if present)
            if raw_id:
                if raw_id in seen_ids:
                    dropped += 1
                    continue
                seen_ids.add(raw_id)

            # Exact text de-dupe
            if text in seen_texts:
                dropped += 1
                continue
            seen_texts.add(text)

            if role_mode == "tech":
                mapped = _map_to_tech_role(role, text)
                if not mapped:
                    dropped += 1
                    continue
                writer.writerow({"text": text, "role": mapped})
            else:
                writer.writerow({"text": text, "role": role})
            kept += 1

    return kept, dropped


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        # Python 3.7+: prefer UTF-8 output when supported.
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default="",
        help="Input raw CSV. Defaults to data new/raw/Resume_extended.csv (or Resume.csv fallback).",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=str,
        default="",
        help="Output CSV with columns: text, role",
    )
    p.add_argument(
        "--role-mode",
        choices=["category", "tech"],
        default="category",
        help="Keep original categories, or map to app-friendly tech roles.",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Drop resumes shorter than this many characters",
    )
    args = p.parse_args()

    in_path = Path(args.in_path) if args.in_path else _pick_input(repo_root)
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = repo_root / "data" / (
            "job_roles_tech_from_resume.csv" if args.role_mode == "tech" else "job_roles_from_resume.csv"
        )

    kept, dropped = prepare(
        in_path,
        out_path,
        min_chars=int(args.min_chars),
        role_mode=str(args.role_mode),
    )

    _safe_print(f"Input:  {in_path}")
    _safe_print(f"Output: {out_path}")
    _safe_print(f"Role mode: {args.role_mode}")
    _safe_print(f"Kept:   {kept}")
    _safe_print(f"Dropped:{dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

