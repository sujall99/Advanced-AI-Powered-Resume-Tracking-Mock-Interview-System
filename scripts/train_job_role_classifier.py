"""
Train and save the Job Role Classifier (TF-IDF + MiniLM + Logistic Regression).

Usage (from repo root):
    python scripts/train_job_role_classifier.py --data data/job_roles_sample.csv --out models/job_role_classifier.joblib

CSV format:
    - Must contain columns: text, role

After training, `app.py` can load the saved model for better predictions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.job_role_classifier import JobRoleClassifier


def load_dataset(path: Path) -> Tuple[List[str], List[str]]:
    X: List[str] = []
    y: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        required = {"text", "role"}
        missing = required - set([h.strip() for h in reader.fieldnames])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            text = (row.get("text") or "").strip()
            role = (row.get("role") or "").strip()
            if not text or not role:
                continue
            X.append(text)
            y.append(role)
    if len(X) < 4:
        raise ValueError("Need at least 4 valid rows to train. Add more labeled samples.")
    return X, y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to CSV with columns: text, role")
    p.add_argument(
        "--out",
        type=str,
        default="models/job_role_classifier.joblib",
        help="Output path for saved model",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable holdout classification report during training.",
    )
    args = p.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    X, y = load_dataset(data_path)

    clf = JobRoleClassifier()
    clf.fit(X, y, evaluate=not bool(args.no_eval))
    clf.save(out_path)

    print(f"Trained on {len(X)} rows.")
    print(f"Saved model to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

