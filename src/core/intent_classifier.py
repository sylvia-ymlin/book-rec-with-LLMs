"""
Model-based intent classifier for Query Router.

Replaces brittle rule-based heuristics with a trained classifier.
Backends: tfidf (default), fasttext, distilbert.

Intents: small_to_big (detail), fast (keyword), deep (natural language)
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

INTENTS = ["small_to_big", "fast", "deep"]


class IntentClassifier:
    """
    Intent classifier with pluggable backends:
    - tfidf: TF-IDF + LogisticRegression (~1–2ms)
    - fasttext: FastText (~1ms, requires fasttext package)
    - distilbert: Zero-shot DistilBERT (~50–100ms, higher accuracy)
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.pipeline: Optional[Pipeline] = None
        self._fasttext_model = None
        self._distilbert_pipeline = None
        self._backend = "tfidf"
        self.model_path = Path(model_path) if model_path else None

    def load(self, path: Optional[Path] = None) -> bool:
        """Load trained model from disk."""
        p = path or self.model_path
        if not p:
            return False
        p = Path(p)
        base = p.parent if p.suffix in (".pkl", ".bin") else p
        pkl_path = p if p.suffix == ".pkl" else base / "intent_classifier.pkl"
        bin_path = p if p.suffix == ".bin" else base / "intent_classifier.bin"

        # Try .pkl first (tfidf or distilbert)
        if pkl_path.exists():
            try:
                data = joblib.load(pkl_path)
                if isinstance(data, dict):
                    self.pipeline = data.get("pipeline")
                    self._backend = data.get("backend", "tfidf")
                    if self._backend == "distilbert":
                        self._load_distilbert(data)
                    elif self.pipeline is None and self._backend == "tfidf":
                        self.pipeline = data
                else:
                    self.pipeline = data
                self.model_path = pkl_path
                logger.info("Intent classifier loaded from %s (backend=%s)", pkl_path, self._backend)
                return True
            except Exception as e:
                logger.warning("Failed to load intent classifier: %s", e)

        # Try .bin (FastText)
        if bin_path.exists():
            try:
                import fasttext
                self._fasttext_model = fasttext.load_model(str(bin_path))
                self._backend = "fasttext"
                self.model_path = bin_path
                logger.info("Intent classifier loaded from %s (FastText)", bin_path)
                return True
            except ImportError:
                logger.warning("FastText not installed; pip install fasttext")
            except Exception as e:
                logger.warning("Failed to load FastText: %s", e)

        return False

    def _load_distilbert(self, data: dict) -> None:
        """Lazy-load DistilBERT pipeline from saved config."""
        model_name = data.get("distilbert_model", "distilbert-base-uncased")
        try:
            from transformers import pipeline
            self._distilbert_pipeline = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,
            )
        except Exception as e:
            logger.warning("DistilBERT pipeline load failed: %s", e)
        self.pipeline = None  # Use distilbert, not sklearn pipeline

    def predict(self, query: str) -> str:
        """Predict intent for a query. Returns one of small_to_big, fast, deep."""
        q = query.strip()
        if not q:
            return "deep"

        if self._fasttext_model is not None:
            pred = self._fasttext_model.predict(q)
            return pred[0][0].replace("__label__", "")

        if self._distilbert_pipeline is not None:
            out = self._distilbert_pipeline(q, INTENTS, multi_label=False)
            return out["labels"][0]

        if self.pipeline is None:
            raise RuntimeError("Intent classifier not loaded; call load() first")
        return str(self.pipeline.predict([q])[0])

    def predict_proba(self, query: str) -> dict[str, float]:
        """Return intent probabilities for debugging."""
        q = query.strip()
        if not q:
            return {i: 1.0 / len(INTENTS) for i in INTENTS}

        if self._fasttext_model is not None:
            pred = self._fasttext_model.predict(q, k=len(INTENTS))
            return dict(zip([l.replace("__label__", "") for l in pred[0]], pred[1]))

        if self._distilbert_pipeline is not None:
            out = self._distilbert_pipeline(q, INTENTS, multi_label=False)
            return dict(zip(out["labels"], out["scores"]))

        if self.pipeline is None:
            raise RuntimeError("Intent classifier not loaded")
        probs = self.pipeline.predict_proba([q])[0]
        last_step = self.pipeline.steps[-1][1]
        classes = getattr(last_step, "classes_", INTENTS)
        return dict(zip(classes, probs))


def train_classifier(
    queries: list[str],
    labels: list[str],
    max_features: int = 5000,
    C: float = 1.0,
    backend: str = "tfidf",
):
    """
    Train intent classifier. Returns pipeline (tfidf), model (fasttext), or dict (distilbert).
    """
    if backend == "fasttext":
        return _train_fasttext(queries, labels)
    if backend == "distilbert":
        return _train_distilbert(queries, labels)
    # tfidf default
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=500,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    pipeline.fit(queries, labels)
    return pipeline


def _train_fasttext(queries: list[str], labels: list[str]):
    """Train FastText classifier. Requires fasttext package."""
    try:
        import fasttext
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for q, l in zip(queries, labels):
                line = q.replace("\n", " ").strip()
                f.write(f"__label__{l} {line}\n")
            path = f.name
        model = fasttext.train_supervised(path, epoch=25, lr=0.5, wordNgrams=2)
        os.unlink(path)
        return model
    except ImportError:
        raise RuntimeError("FastText not installed: pip install fasttext")


def _train_distilbert(queries: list[str], labels: list[str]) -> dict:
    """DistilBERT zero-shot: creates pipeline (no training). Saves config for inference."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "zero-shot-classification",
            model="distilbert-base-uncased",
            device=-1,
        )
        return {
            "backend": "distilbert",
            "distilbert_model": "distilbert-base-uncased",
            "intents": INTENTS,
        }
    except Exception as e:
        raise RuntimeError(f"DistilBERT setup failed: {e}")
