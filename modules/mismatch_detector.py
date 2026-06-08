"""
Resume ↔ Job Description mismatch detection.

Goal:
Show a warning when the candidate resume domain is far from the job description domain.

We use two lightweight signals:
1) TF-IDF cosine similarity between resume text and JD text
2) Role/domain mismatch between resume-only prediction and JD-only prediction (optional)

This is intentionally simple and dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class MismatchResult:
    similarity: float  # 0..1 cosine similarity
    is_mismatch: bool
    reason: str


def tfidf_similarity(resume_text: str, jd_text: str) -> float:
    a = (resume_text or "").strip()
    b = (jd_text or "").strip()
    if not a or not b:
        return 0.0

    # Fit on just the two docs (fast, no external corpus).
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([a, b]).toarray()
    v1, v2 = X[0], X[1]
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
    return float(np.dot(v1, v2) / denom)


def detect_mismatch(
    resume_text: str,
    jd_text: str,
    *,
    similarity_threshold: float = 0.12,
    resume_role: Optional[str] = None,
    jd_role: Optional[str] = None,
) -> MismatchResult:
    """
    Returns a mismatch warning when:
    - Similarity between resume and JD is very low AND
    - Roles look different (when provided)
    """
    sim = tfidf_similarity(resume_text, jd_text)

    if not (jd_text or "").strip():
        return MismatchResult(similarity=sim, is_mismatch=False, reason="No job description provided.")

    # If roles are available and clearly different, we rely on similarity to decide mismatch.
    roles_differ = bool(resume_role and jd_role and resume_role != jd_role)

    if sim < similarity_threshold and roles_differ:
        return MismatchResult(
            similarity=sim,
            is_mismatch=True,
            reason="Resume does not match JD (domain mismatch).",
        )

    if sim < (similarity_threshold / 2):
        # Extreme mismatch even without role info
        return MismatchResult(
            similarity=sim,
            is_mismatch=True,
            reason="Resume appears very different from the JD (low similarity).",
        )

    return MismatchResult(similarity=sim, is_mismatch=False, reason="Resume and JD appear reasonably aligned.")

