# tests/test_smoke.py
"""
Smoke tests that validate the core pipeline logic used across notebooks 05 and app/app.py.
Uses a sampled in-memory dataset — no file I/O, no heavy deps like torch/prophet.
"""
import re
import pickle
import pathlib
import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ── helpers mirrored from the notebooks ───────────────────────────────────────

SENTIMENT_MAP = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}

NEGATION_PATTERNS = [
    (r"\bnot good\b", "negative"),
    (r"\bnot working\b", "negative"),
    (r"\bhakuna network\b", "negative"),
    (r"\bsi nzuri\b", "negative"),
]


def derive_sentiment(score: int) -> str:
    return SENTIMENT_MAP.get(int(score), "Neutral")


def apply_negation(text: str) -> str:
    for pattern, replacement in NEGATION_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


# ── tests ──────────────────────────────────────────────────────────────────────

class TestDataSchema:
    """Notebook 02 / 03 – schema & feature engineering checks."""

    def test_required_columns_present(self, sample_df):
        required = {
            "content", "score", "app_name", "sentiment_label",
            "complaint_label", "fraud_indicator", "cleaned_text",
            "processed_text", "final_language",
        }
        assert required.issubset(set(sample_df.columns)), \
            f"Missing columns: {required - set(sample_df.columns)}"

    def test_score_range(self, sample_df):
        assert sample_df["score"].between(1, 5).all(), "Scores must be 1-5"

    def test_sentiment_labels_valid(self, sample_df):
        valid = {"Positive", "Neutral", "Negative"}
        assert set(sample_df["sentiment_label"]).issubset(valid)

    def test_no_null_processed_text(self, sample_df):
        assert sample_df["processed_text"].notna().all()

    def test_fraud_indicator_is_bool(self, sample_df):
        assert sample_df["fraud_indicator"].dtype == bool or \
               set(sample_df["fraud_indicator"].unique()).issubset({True, False, 0, 1})


class TestPreprocessing:
    """Notebook 03 – text cleaning and negation handling."""

    def test_derive_sentiment_mapping(self):
        assert derive_sentiment(1) == "Negative"
        assert derive_sentiment(3) == "Neutral"
        assert derive_sentiment(5) == "Positive"

    def test_basic_clean_removes_urls(self):
        result = basic_clean("check https://example.com now")
        assert "http" not in result

    def test_basic_clean_lowercases(self):
        assert basic_clean("M-PESA") == "mpesa"

    def test_negation_handler_english(self):
        assert apply_negation("not good app") == "negative app"

    def test_negation_handler_swahili(self):
        assert apply_negation("hakuna network leo") == "negative leo"

    def test_sentiment_on_sampled_data(self, sample_df):
        derived = sample_df["score"].apply(derive_sentiment)
        assert derived.isin(["Positive", "Neutral", "Negative"]).all()


class TestTfidfPipeline:
    """Notebook 05 – lightweight TF-IDF + classifier round-trip."""

    def test_tfidf_fit_transform(self, sample_df):
        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(sample_df["processed_text"])
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] <= 50

    def test_classifier_trains_and_predicts(self, sample_df):
        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(sample_df["processed_text"])
        y = sample_df["sentiment_label"]
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(sample_df)
        assert set(preds).issubset({"Positive", "Neutral", "Negative"})


class TestModelArtifacts:
    """Verify production model pickle can be loaded and returns predictions."""

    MODEL_PATH = pathlib.Path("app/advancedxgboostmodelfinal.pkl")

    @pytest.mark.skip(reason="Model artifact skipped — too large for CI and may be Git LFS pointer locally")
    def test_model_loads(self):
        with open(self.MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        assert model is not None

    @pytest.mark.skip(reason="Model artifact skipped — too large for CI and may be Git LFS pointer locally")
    def test_model_predicts(self):
        with open(self.MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        result = pipeline.predict(["mpesa not working failed transaction"])
        assert result[0] in ["Positive", "Neutral", "Negative", 0, 1, 2]


class TestFDI:
    """Notebook 07 – Financial Distress Index computation."""

    def test_fdi_score_range(self, sample_df):
        """FDI should be normalised to [0, 1]."""
        df = sample_df.copy()
        df["is_negative"] = df["sentiment_label"] == "Negative"
        df["is_fraud"] = df["fraud_indicator"].astype(bool)

        complaint_rate = df["is_negative"].mean()
        fraud_rate = df["is_fraud"].mean()
        distress_score = (fraud_rate * 0.6) + (complaint_rate * 0.4)

        assert 0.0 <= distress_score <= 1.0, \
            f"FDI out of range: {distress_score}"

    def test_distress_level_classification(self):
        thresholds = [(0.0, "Green"), (0.25, "Yellow"), (0.5, "Orange"), (0.75, "Red")]

        def classify(score):
            if score < 0.25:
                return "Green"
            if score < 0.5:
                return "Yellow"
            if score < 0.75:
                return "Orange"
            return "Red"

        for score, expected in thresholds:
            assert classify(score) == expected