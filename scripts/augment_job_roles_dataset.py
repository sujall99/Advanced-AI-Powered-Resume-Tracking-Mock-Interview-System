"""
Dataset augmentation (template-based).

This script helps you reach 50–200 samples per role quickly without needing any API.
It creates synthetic variations by combining common responsibilities + skills per role.

Usage:
  python scripts/augment_job_roles_dataset.py --in data/job_roles_sample.csv --out data/job_roles_augmented.csv --per-role 60

Notes:
  - This is a baseline. Real labeled data (real resumes/JDs) still beats synthetic.
  - Use this to reduce "low confidence" due to tiny datasets.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROLE_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "AI/Data Science Engineer": {
        "skills": ["Python", "SQL", "Flask", "NLP", "machine learning", "model evaluation", "LLMs", "IoT"],
        "tasks": ["build AI prototypes", "deploy lightweight APIs", "validate text outputs", "integrate sensor data"],
        "tools": ["ThingSpeak", "ESP32", "scikit-learn", "pandas"],
    },
    "ML Engineer": {
        "skills": ["Python", "scikit-learn", "model deployment", "feature engineering", "monitoring"],
        "tasks": ["train models", "serve models via API", "track experiments", "evaluate metrics"],
        "tools": ["Flask", "FastAPI", "MLflow", "Docker"],
    },
    "Data Scientist": {
        "skills": ["Python", "pandas", "statistics", "machine learning", "EDA"],
        "tasks": ["analyze data", "build predictive models", "create notebooks", "communicate insights"],
        "tools": ["scikit-learn", "Jupyter", "matplotlib", "seaborn"],
    },
    "Data Engineer": {
        "skills": ["SQL", "ETL", "data modeling", "pipelines", "data quality"],
        "tasks": ["build ETL pipelines", "orchestrate workflows", "optimize queries", "monitor jobs"],
        "tools": ["Airflow", "Spark", "Kafka", "S3"],
    },
    "Backend Developer": {
        "skills": ["Python", "REST APIs", "databases", "testing", "auth"],
        "tasks": ["build APIs", "design schemas", "write tests", "deploy services"],
        "tools": ["FastAPI", "Django", "PostgreSQL", "Docker"],
    },
    "Frontend Developer": {
        "skills": ["React", "TypeScript", "CSS", "accessibility", "UI"],
        "tasks": ["build components", "optimize performance", "improve UX", "integrate APIs"],
        "tools": ["Next.js", "Jest", "Storybook", "Tailwind"],
    },
    "Full Stack Developer": {
        "skills": ["React", "Node.js", "REST", "databases", "auth"],
        "tasks": ["build full stack apps", "integrate frontend/backend", "deploy", "write tests"],
        "tools": ["Express", "PostgreSQL", "Docker", "JWT"],
    },
    "DevOps Engineer": {
        "skills": ["Linux", "CI/CD", "Docker", "Kubernetes", "monitoring"],
        "tasks": ["build pipelines", "manage deployments", "monitor systems", "incident response"],
        "tools": ["Terraform", "Prometheus", "Grafana", "GitHub Actions"],
    },
    "Chemical Engineer": {
        "skills": ["process engineering", "unit operations", "safety", "quality", "simulation"],
        "tasks": ["optimize processes", "perform mass/energy balances", "prepare PFD/P&ID", "run HAZOP"],
        "tools": ["Aspen HYSYS", "Excel", "lab testing"],
    },
    "Electrical Engineer": {
        "skills": ["circuits", "power systems", "control systems", "troubleshooting", "PLC basics"],
        "tasks": ["design electrical systems", "prepare SLDs", "test and debug", "maintain equipment"],
        "tools": ["MATLAB", "SCADA", "AutoCAD Electrical"],
    },
    "Mechanical Engineer": {
        "skills": ["CAD", "machine design", "manufacturing", "thermodynamics", "maintenance"],
        "tasks": ["design parts", "create drawings", "prototype", "analyze failures"],
        "tools": ["SolidWorks", "AutoCAD", "ANSYS"],
    },
    "Civil Engineer": {
        "skills": ["structural design", "construction management", "estimation", "surveying", "QA/QC"],
        "tasks": ["supervise sites", "prepare BOQ", "coordinate contractors", "ensure compliance"],
        "tools": ["AutoCAD", "STAAD.Pro", "MS Project"],
    },
    "Electronics Engineer": {
        "skills": ["analog/digital electronics", "PCB basics", "signals", "communication systems", "IoT"],
        "tasks": ["test circuits", "debug hardware", "document results", "prototype devices"],
        "tools": ["oscilloscope", "Arduino", "ESP32"],
    },
}


def read_csv(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = (r.get("text") or "").strip()
            role = (r.get("role") or "").strip()
            if t and role:
                rows.append((t, role))
    return rows


def write_csv(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "role"])
        w.writeheader()
        for t, role in rows:
            w.writerow({"text": t, "role": role})


def synthesize(role: str, n: int, seed: int) -> List[Tuple[str, str]]:
    rnd = random.Random(seed)
    tpl = ROLE_TEMPLATES.get(role)
    if not tpl:
        return []
    out: List[Tuple[str, str]] = []
    for _ in range(n):
        skills = rnd.sample(tpl["skills"], k=min(4, len(tpl["skills"])))
        tasks = rnd.sample(tpl["tasks"], k=min(2, len(tpl["tasks"])))
        tools = rnd.sample(tpl["tools"], k=min(2, len(tpl["tools"])))
        text = (
            f"{role} experience: {', '.join(tasks)}. Skills: {', '.join(skills)}. Tools: {', '.join(tools)}."
        )
        out.append((text, role))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Input CSV (text,role)")
    p.add_argument("--out", required=True, help="Output CSV (text,role)")
    p.add_argument("--per-role", type=int, default=60, help="Target samples per role")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base = read_csv(Path(args.inp))
    by_role: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for t, role in base:
        by_role[role].append((t, role))

    augmented: List[Tuple[str, str]] = list(base)
    for role, rows in sorted(by_role.items()):
        need = max(0, int(args.per_role) - len(rows))
        augmented.extend(synthesize(role, need, seed=args.seed + hash(role) % 10_000))

    # Also synthesize for roles that exist in templates but not in CSV yet
    for role in sorted(ROLE_TEMPLATES.keys()):
        if role not in by_role:
            augmented.extend(synthesize(role, int(args.per_role), seed=args.seed + hash(role) % 10_000))

    write_csv(Path(args.out), augmented)
    print(f"Wrote {len(augmented)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

