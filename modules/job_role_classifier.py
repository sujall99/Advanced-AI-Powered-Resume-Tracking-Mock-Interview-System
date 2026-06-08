"""
Job Role Classification Module
==============================

Hybrid resume role classifier with:
- TF-IDF sparse lexical features
- BERT-style sentence embeddings (`all-MiniLM-L6-v2`)
- Concatenated feature vectors (`numpy.hstack`)
- Balanced classifier training for imbalanced labels

Design goals:
- Preserve Streamlit-compatible API (`predict`, `save`, `load`, `load_or_demo`)
- Keep training configurable and reusable
- Normalize/merge rare labels before training
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple

import joblib
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class RolePrediction:
    predicted_role: str
    confidence: float
    top_k: List[Dict[str, float]]
    rejected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_role": self.predicted_role,
            "confidence": float(self.confidence),
            "top_k": [{"role": x["role"], "confidence": float(x["confidence"])} for x in self.top_k],
            "rejected": bool(self.rejected),
        }


class JobRoleClassifier:
    """
    Hybrid (TF-IDF + MiniLM embeddings) job-role classifier.

    Public API remains stable for Streamlit:
    - `fit(...)`
    - `predict(...)`
    - `save(...)` / `load(...)`
    """

    def __init__(
        self,
        *,
        max_features: int = 30_000,
        ngram_range: Tuple[int, int] = (1, 2),
        lowercase: bool = True,
        # With small/medium custom datasets, min_df=1 is safer (prevents dropping useful terms).
        min_df: int = 1,
        C: float = 3.0,
        random_state: int = 42,
        bert_model_name: str = "all-MiniLM-L6-v2",
        role_merge_map: Optional[Dict[str, str]] = None,
        rare_class_threshold: int = 5,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=lowercase,
            min_df=min_df,
            stop_words="english",
        )
        # Balanced loss reduces class-imbalance bias during optimization.
        self.classifier = LogisticRegression(
            max_iter=2000,
            C=C,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=None,
            random_state=random_state,
        )
        self.label_encoder = LabelEncoder()
        self.bert_model_name = bert_model_name
        self.role_merge_map = role_merge_map or {
            "AI Engineer": "Machine Learning Engineer",
            "ML Engineer": "Machine Learning Engineer",
            "Data Scientist": "Data Science",
            "Data Analyst": "Data Analysis",
        }
        self.rare_class_threshold = int(rare_class_threshold)
        self._embedder = None
        self._is_fitted = False
        # Compatibility field for older saved payloads.
        self.pipeline = None

    # -----------------------------
    # Training
    # -----------------------------
    def fit(
        self,
        X: List[str],
        y: List[str],
        *,
        evaluate: bool = True,
        test_size: float = 0.2,
    ) -> "JobRoleClassifier":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(X) < 4:
            raise ValueError("Need at least 4 samples to fit a meaningful baseline classifier.")

        # 1) Show raw label distribution and highlight rare classes.
        self._print_class_distribution("Before merge", y)
        rare_before = self._find_rare_classes(y, threshold=self.rare_class_threshold)
        if rare_before:
            print(f"[INFO] Rare classes (<{self.rare_class_threshold} samples): {rare_before}")

        # 2) Normalize labels with configurable semantic merge map.
        y_norm = self.normalize_labels(y)
        self._print_class_distribution("After merge", y_norm)
        rare_after = self._find_rare_classes(y_norm, threshold=self.rare_class_threshold)
        if rare_after:
            print(
                f"[WARN] Still rare after merge (<{self.rare_class_threshold} samples): "
                f"{rare_after}. Consider extending mapping or collecting more data."
            )

        if len(set(y_norm)) < 2:
            raise ValueError("Need at least 2 classes after label normalization.")

        # 3) Build hybrid features: TF-IDF + MiniLM sentence embeddings.
        X_features = self._build_features(X, fit=True)
        y_encoded = self.label_encoder.fit_transform(y_norm)

        # 4) Optional evaluation with classification report (not accuracy-only).
        can_stratify = min(Counter(y_norm).values()) >= 2
        can_split = evaluate and len(X) >= 10 and can_stratify
        if can_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features,
                y_encoded,
                test_size=float(test_size),
                random_state=42,
                stratify=y_encoded,
            )
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            target_names = self.label_encoder.inverse_transform(np.unique(y_test))
            print("\nClassification report:")
            print(
                classification_report(
                    y_test,
                    y_pred,
                    labels=np.unique(y_test),
                    target_names=list(target_names),
                    zero_division=0,
                )
            )

            # Refit on full dataset for production use after evaluation snapshot.
            self.classifier.fit(X_features, y_encoded)
        else:
            self.classifier.fit(X_features, y_encoded)

        self._is_fitted = True
        return self

    def fit_default_demo_dataset(self) -> "JobRoleClassifier":
        """
        Fit a tiny built-in dataset so the module works immediately.

        Replace this with your own labeled data for real quality.
        """
        X, y = _default_demo_dataset()
        if getattr(self.vectorizer, "min_df", 2) > 1:
            self.vectorizer.set_params(min_df=1)
        return self.fit(X, y, evaluate=False)

    @staticmethod
    def default_model_path() -> Path:
        """
        Default on-disk model location used by the Streamlit app.
        """
        return Path("models") / "job_role_classifier.joblib"

    @classmethod
    def load_or_demo(cls, path: str | Path | None = None) -> "JobRoleClassifier":
        """
        Load a trained model from disk if present; otherwise fall back to demo training.
        """
        model_path = Path(path) if path is not None else cls.default_model_path()
        if model_path.exists():
            return cls.load(model_path)
        return cls().fit_default_demo_dataset()

    # -----------------------------
    # Inference
    # -----------------------------
    def predict(
        self,
        text: str,
        *,
        top_k: int = 3,
        reject_below: float = 0.20,
        reject_label: str = "Unknown/Other",
    ) -> Dict[str, Any]:
        """
        Return:
        - predicted_role
        - confidence (0..1)
        - top_k list of roles with confidences
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Classifier is not fitted. Call `fit(...)` or `fit_default_demo_dataset()` first."
            )
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string.")

        probs = self._predict_proba(text)
        classes = list(self._classes())

        idx = int(np.argmax(probs))
        predicted = classes[idx]
        confidence = float(probs[idx])

        k = max(1, min(int(top_k), len(classes)))
        top_indices = np.argsort(probs)[::-1][:k]
        top_list = [{"role": classes[i], "confidence": float(probs[i])} for i in top_indices]

        if confidence < float(reject_below):
            return RolePrediction(reject_label, confidence, top_list, rejected=True).to_dict()

        return RolePrediction(predicted, confidence, top_list, rejected=False).to_dict()

    def _predict_proba(self, text: str) -> np.ndarray:
        # Backward compatibility path for older saved pipeline-only models.
        if self.pipeline is not None:
            model = self.pipeline
            if not hasattr(model, "predict_proba"):
                raise RuntimeError("Underlying model does not support probability predictions.")
            proba = model.predict_proba([text])[0]
            return np.asarray(proba, dtype=float)

        features = self._build_features([text], fit=False)
        if not hasattr(self.classifier, "predict_proba"):
            raise RuntimeError("Underlying model does not support probability predictions.")
        proba = self.classifier.predict_proba(features)[0]
        return np.asarray(proba, dtype=float)

    def _classes(self) -> np.ndarray:
        # Backward compatibility path for older saved pipeline-only models.
        if self.pipeline is not None:
            if not hasattr(self.pipeline, "classes_"):
                raise RuntimeError("Underlying model has no classes_.")
            return self.pipeline.classes_
        return np.asarray(self.label_encoder.classes_)

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, path: str | Path) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "label_encoder": self.label_encoder,
                "bert_model_name": self.bert_model_name,
                "role_merge_map": self.role_merge_map,
                "rare_class_threshold": self.rare_class_threshold,
                "is_fitted": self._is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "JobRoleClassifier":
        path = Path(path)
        payload = joblib.load(path)
        obj = cls()

        # New hybrid payload.
        if "vectorizer" in payload and "classifier" in payload:
            obj.vectorizer = payload["vectorizer"]
            obj.classifier = payload["classifier"]
            obj.label_encoder = payload["label_encoder"]
            obj.bert_model_name = payload.get("bert_model_name", obj.bert_model_name)
            obj.role_merge_map = payload.get("role_merge_map", obj.role_merge_map)
            obj.rare_class_threshold = int(
                payload.get("rare_class_threshold", obj.rare_class_threshold)
            )
            obj._is_fitted = bool(payload.get("is_fitted", True))
            obj.pipeline = None
            return obj

        # Legacy payload fallback to avoid breaking older saved models.
        obj.pipeline = payload.get("pipeline")
        obj._is_fitted = bool(payload.get("is_fitted", True))
        return obj

    # -----------------------------
    # Label normalization helpers
    # -----------------------------
    def normalize_labels(self, labels: List[str]) -> List[str]:
        """Apply configurable semantic role mapping before training."""
        return [self.role_merge_map.get(lbl, lbl) for lbl in labels]

    @staticmethod
    def _find_rare_classes(labels: List[str], threshold: int = 5) -> Dict[str, int]:
        counts = Counter(labels)
        return {k: v for k, v in sorted(counts.items(), key=lambda x: (x[1], x[0])) if v < threshold}

    @staticmethod
    def _print_class_distribution(title: str, labels: List[str]) -> None:
        counts = Counter(labels)
        print(f"\nClass distribution ({title}):")
        for role, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {role}: {cnt}")

    # -----------------------------
    # Feature builders
    # -----------------------------
    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise RuntimeError(
                    "sentence-transformers is required for hybrid role classification. "
                    "Install dependencies from requirements.txt."
                ) from exc
            self._embedder = SentenceTransformer(self.bert_model_name)
        return self._embedder

    def _build_features(self, texts: List[str], *, fit: bool) -> np.ndarray:
        if fit:
            tfidf_sparse = self.vectorizer.fit_transform(texts)
        else:
            tfidf_sparse = self.vectorizer.transform(texts)

        tfidf_dense = tfidf_sparse.toarray().astype(np.float32, copy=False)
        bert_dense = self._get_embedder().encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

        # Explicitly use numpy.hstack as requested for feature fusion.
        return np.hstack([tfidf_dense, bert_dense])


def _default_demo_dataset() -> Tuple[List[str], List[str]]:
    """
    Very small dataset for immediate local demos.

    Replace with your own labeled data for real use.
    """
    samples: List[Tuple[str, str]] = [
        (
            "Built REST APIs with Python, Django, PostgreSQL. Deployed on AWS. Wrote unit tests.",
            "Backend Developer",
        ),
        (
            "React, TypeScript, Next.js, CSS, UI components, accessibility, frontend performance.",
            "Frontend Developer",
        ),
        (
            "ETL pipelines, Spark, Airflow, SQL, data lakes, AWS Glue, Kafka, batch processing.",
            "Data Engineer",
        ),
        (
            "Machine learning, pandas, scikit-learn, model evaluation, feature engineering, Python.",
            "Data Science",
        ),
        (
            "Business dashboards, Power BI, Tableau, SQL reporting, stakeholder communication.",
            "Data Analysis",
        ),
        (
            "Kubernetes, Docker, CI/CD, Terraform, monitoring, SRE, Linux, incident response.",
            "DevOps Engineer",
        ),
        (
            "Java, Spring Boot microservices, Kafka, Redis, MySQL, scalability, distributed systems.",
            "Backend Developer",
        ),
        (
            "Node.js, Express, React, MongoDB, full stack web apps, REST, authentication, JWT.",
            "Full Stack Developer",
        ),
        (
            "Fine-tuned transformer models, built retrieval systems, model deployment APIs.",
            "Machine Learning Engineer",
        ),
    ]
    X = [s[0] for s in samples]
    y = [s[1] for s in samples]
    return X, y

