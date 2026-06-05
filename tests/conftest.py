# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def sample_df():
    """20-row sampled dataset that mirrors the real cleaned_data.csv schema."""
    rng = np.random.default_rng(42)
    n = 20
    apps = ["M-PESA", "Tala", "Branch", "KCB Mobile", "Equity Mobile", "MySafaricom"]
    sentiments = ["Positive", "Neutral", "Negative"]
    complaints = ["Fraud Complaint", "Failed Transaction", "Hidden Charges",
                  "Customer Support", "General"]

    df = pd.DataFrame({
        "content": [f"Sample review number {i}" for i in range(n)],
        "score": rng.integers(1, 6, size=n),
        "app_name": rng.choice(apps, size=n),
        "sentiment_label": rng.choice(sentiments, size=n),
        "complaint_label": rng.choice(complaints, size=n),
        "fraud_indicator": rng.choice([True, False], size=n),
        "cleaned_text": [f"cleaned review {i}" for i in range(n)],
        "processed_text": [f"processed review {i}" for i in range(n)],
        "final_language": rng.choice(["en", "sw", "mixed"], size=n),
        "review_length": rng.integers(10, 200, size=n),
        "word_count": rng.integers(2, 40, size=n),
    })
    return df