# RELEASE.md

# Pesa Salama Model Release and Data Governance Policy

## 1. Project Overview

Pesa Salama is a Natural Language Processing (NLP) system developed to analyze customer feedback within Kenya's mobile money ecosystem. The project uses machine learning to classify sentiment, identify complaint patterns, and generate a Financial Distress Index (FDI) for major fintech applications including M-Pesa, MySafaricom, KCB Mobile, Equity Mobile, Tala, and Branch.

The objective is to transform large volumes of unstructured customer reviews into actionable insights that can support service improvement, consumer protection initiatives, and data-driven decision-making.

---

## 2. Model Release Process

The project follows a version-controlled release workflow using Git and GitHub.

Each model release includes:

* Updated source code and notebooks
* Trained model artifacts (`.pkl` files)
* Updated documentation
* Performance evaluation reports
* Validation against a held-out test dataset

Model updates are accepted only after confirming that key evaluation metrics remain stable or improve relative to previous versions.

Performance is primarily assessed using:

* Weighted F1 Score
* Precision
* Recall
* Confusion Matrix
* Five-Fold Cross Validation

Although multiple machine learning models were evaluated during development, the deployed version uses the tuned TF-IDF + XGBoost pipeline because it provides an effective balance between predictive accuracy, inference speed, and computational efficiency for local deployment and Streamlit applications.

---

## 2.1 Versioning Strategy

The project follows Semantic Versioning (SemVer):

MAJOR.MINOR.PATCH

- MAJOR: Significant changes to model architecture or data processing pipeline.
- MINOR: New features, improved preprocessing, or model enhancements.
- PATCH: Bug fixes, documentation updates, or minor corrections.

Current Release:

v1.0.0 – Initial release of the Pesa Salama sentiment analysis system, including data preprocessing, model training, evaluation, Financial Distress Index (FDI) generation, and Streamlit deployment.
---

## 2.2 Reproducibility Requirements

Each release should contain sufficient information to reproduce results and validate model performance.

Release artifacts should include:

* Training dataset version
* Feature engineering pipeline
* Model hyperparameters
* Evaluation metrics
* Random seed configuration (where applicable)
* Trained model artifact (`.pkl`)
* Release notes documenting major changes

This ensures transparency, reproducibility, and traceability across model versions.

---

## 2.3 Release Documentation

Every GitHub release should contain:

* Release notes summarizing changes
* Updated evaluation metrics
* Trained model artifacts
* Documentation updates
* Known limitations
* Planned future improvements

Git tags should follow the format:

```text
vMAJOR.MINOR.PATCH
```

Example:

```text
v1.0.0
```

---

## 3. Data Collection and Retention Policy

The project uses publicly available Google Play Store reviews collected through the `google-play-scraper` library.

The retained dataset contains only information necessary for analysis, including:

* Review text
* Application name
* Star rating
* Review timestamp
* Derived sentiment labels

No usernames, email addresses, phone numbers, account identifiers, or other personally identifiable information (PII) are intentionally collected or stored.

The dataset is retained solely for:

* Academic research
* Model training
* Model evaluation
* Performance monitoring
* Reproducibility of results

---

## 3.1 Data Retention Period

The project follows a data minimization approach and retains data only for legitimate research and development purposes.

Retention guidelines include:

* Raw review datasets: retained for the duration of the project
* Processed datasets: retained for reproducibility and future evaluation
* Evaluation reports: retained for model comparison and auditing
* Trained model artifacts: retained indefinitely within release history

Future datasets may replace or refresh historical data while maintaining the same privacy and governance standards.

---

## 4. Privacy Considerations

Pesa Salama is designed to respect user privacy and promote responsible use of publicly available information.

Key privacy principles include:

* Use only publicly accessible review data
* Do not attempt to identify, profile, or track individual users
* Do not combine review data with external personal datasets
* Process user-generated content solely for research, analysis, and demonstration purposes
* Avoid storing unnecessary user information

The Streamlit demonstration application performs real-time predictions on user input. Submitted text is processed temporarily for inference and is not permanently stored, logged, or shared by the application.

---

## 5. Ethical Use and Limitations

The system is intended to provide decision-support insights rather than definitive conclusions.

Sentiment and complaint classifications may occasionally be affected by:

* Multilingual expressions
* Code-switching between English, Swahili, and Sheng
* Sarcasm and informal language
* Ambiguous or context-dependent statements

As a result, model outputs should complement, rather than replace, human judgment when informing operational, business, or regulatory decisions.

The project also acknowledges that machine learning systems may inherit biases present in training data. Continuous evaluation and monitoring are encouraged to identify and mitigate such issues.

---

## 6. Future Governance

Future releases may incorporate:

* Additional fintech applications
* Expanded multilingual support
* Transformer-based language models
* Automated monitoring and reporting pipelines
* Enhanced explainability and interpretability features

Any future expansion of the project will continue to prioritize:

* Transparency
* Reproducibility
* Responsible AI practices
* Data minimization
* User privacy protection

---

## 7. Maintainer

**Project:** Pesa Salama – Mobile Money Customer Sentiment Analysis

**Maintainers:**
1. Marian Amondi
2. Daniel Owuor
3. Brenda Chepkemoi
4. Abby Stacy
5. Angela Mutiga
6. Sonia Cherop
                

This document serves as the project's release management, data retention, and privacy governance policy and should be reviewed periodically as the project evolves.
