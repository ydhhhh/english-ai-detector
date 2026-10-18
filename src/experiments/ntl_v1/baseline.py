"""Transparent, training-only style n-gram baseline for Name That LLM.

This is a compact reimplementation of the four feature channels in the public
stylegram baseline. It uses at most 160,000 features, compared with the public
notebook's 540,000, and does not fit vocabulary or IDF on validation/test text.
The four blocks are separately L2-normalized by TF-IDF, weighted, concatenated,
and passed to one multiclass LinearSVC.

``probabilities`` converts SVM margins to soft targets. These are NOT calibrated
probabilities unless temperature is fitted using a separate validation set or
training-only out-of-fold predictions. The caller owns all split/calibration
decisions so that no validation labels can enter this estimator accidentally.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

import numpy as np
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_consistent_length, check_is_fitted


FUNCTION_WORDS = frozenset(
    "a an the and or but if while although because as so than that which who "
    "whom whose this these those it its they them their theirs he him his she "
    "her hers we us our ours you your yours i me my mine in on at to from with "
    "without for of by over under between through into about is are was were "
    "be been being have has had do does did can could will would shall should "
    "may might must not no nor all any some each every both either neither "
    "other another such only also just very too more most less least here "
    "there when where why how what whether then now still yet even much many "
    "few little".split()
)

# name, analyzer, n-gram range, feature limit, block weight.
FEATURE_SPECS = (
    ("char", "char", (3, 5), 80_000, 1.0),
    ("word", "word", (1, 3), 45_000, 0.7),
    ("shape", "char", (2, 5), 20_000, 0.5),
    ("function", "word", (1, 3), 15_000, 0.5),
)


def shape_text(text: str) -> str:
    """Collapse alphabetic runs to x/X, preserving punctuation and digits.

    X denotes an all-uppercase run; title-case and lowercase runs become x.
    This intentionally does not encode word length, matching the reference
    channel's broad distinction between lexical content and textual shape.
    """
    return re.sub(r"[^\W\d_]+", lambda match: "X" if match[0].isupper() else "x", text)


def function_text(text: str) -> str:
    """Keep function words/punctuation and replace content words by CONTENT."""
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return " ".join(
        token.lower()
        if token.lower() in FUNCTION_WORDS or not token.isalnum()
        else "CONTENT"
        for token in tokens
    )


def probabilities(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax SVM margins with a positive temperature; not auto-calibrated.

    Columns follow ``StyleBaseline.classes_``. For sklearn's binary 1-D margin
    output, create columns [-margin, margin] in the estimator's class order.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = np.column_stack((-values, values))
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must have shape (n_samples, n_classes >= 2)")
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    return softmax(values / temperature, axis=1)


class StyleBaseline(ClassifierMixin, BaseEstimator):
    """Four-channel TF-IDF + LinearSVC; serializable with pickle/joblib.

    Parameters are deliberately few and fixed by default. Vocabulary selection,
    IDF statistics and the SVM are all fitted only to the supplied ``texts``.
    In addition to standard ``classes_``, fitted estimators expose
    ``feature_counts_``, ``features_count``, ``fit_seconds_`` and ``stats_info``.
    """

    def __init__(
        self,
        C: float = 0.5,
        min_df: int = 2,
        max_iter: int = 5000,
        random_state: int = 42,
    ):
        self.C = C
        self.min_df = min_df
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _prepare(texts: Sequence[str], channel: str) -> list[str]:
        if channel == "shape":
            return [shape_text(text) for text in texts]
        if channel == "function":
            return [function_text(text) for text in texts]
        return list(texts)

    @staticmethod
    def _text_list(texts: Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not one string")
        result = list(texts)
        if not all(isinstance(text, str) for text in result):
            raise TypeError("all texts must be strings")
        return result

    def fit(self, texts: Sequence[str], y: Sequence) -> "StyleBaseline":
        started = time.perf_counter()
        texts = self._text_list(texts)
        check_consistent_length(texts, y)
        self.vectorizers_ = {}
        self.feature_counts_ = {}
        blocks = []
        for channel, analyzer, ngrams, limit, weight in FEATURE_SPECS:
            options = dict(
                analyzer=analyzer,
                ngram_range=ngrams,
                lowercase=False,
                sublinear_tf=True,
                min_df=self.min_df,
                max_features=limit,
                dtype=np.float32,
            )
            if analyzer == "word":
                options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
            vectorizer = TfidfVectorizer(**options)
            block = vectorizer.fit_transform(self._prepare(texts, channel))
            self.vectorizers_[channel] = vectorizer
            self.feature_counts_[channel] = int(block.shape[1])
            blocks.append(block * weight)
        matrix = hstack(blocks, format="csr", dtype=np.float32)
        self.features_count = int(matrix.shape[1])
        self.n_features_ = self.features_count
        self.train_nnz_ = int(matrix.nnz)
        self.train_samples_ = len(texts)
        # Explicit dual=True is portable across sklearn versions and is suited
        # to this many-features/few-documents linear classification problem.
        self.classifier_ = LinearSVC(
            C=self.C,
            dual=True,
            random_state=self.random_state,
            max_iter=self.max_iter,
        ).fit(matrix, y)
        self.classes_ = self.classifier_.classes_
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def transform(self, texts: Sequence[str]):
        """Transform using training vocabularies and IDF, without updating them."""
        check_is_fitted(self, "classifier_")
        texts = self._text_list(texts)
        blocks = [
            self.vectorizers_[channel].transform(self._prepare(texts, channel)) * weight
            for channel, _, _, _, weight in FEATURE_SPECS
        ]
        return hstack(blocks, format="csr", dtype=np.float32)

    def decision_function(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier_.decision_function(self.transform(texts))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier_.predict(self.transform(texts))

    def stats_info(self) -> dict:
        """Small JSON-compatible record for experiment reports."""
        check_is_fitted(self, "classifier_")
        return {
            "feature_counts": dict(self.feature_counts_),
            "features_count": self.features_count,
            "train_samples": self.train_samples_,
            "train_nnz": self.train_nnz_,
            "classes_count": len(self.classes_),
            "fit_seconds": self.fit_seconds_,
            "solver_iterations": int(np.max(self.classifier_.n_iter_)),
            "C": self.C,
            "min_df": self.min_df,
            "block_weights": {name: weight for name, _, _, _, weight in FEATURE_SPECS},
        }
