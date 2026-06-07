import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.model_selection import cross_val_score
"""
## Model Evaluation Function

- A reusable model evaluation function was created to train and assess machine learning models using multiple performance metrics. The function calculates the weighted F1 score, precision, recall, classification report, confusion matrix, and cross-validation score to measure model performance and reliability.
"""

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name='Model'):
    
    # 1. Train and Predict
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 2. Calculate Metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. Cross-Validation (Keep the array of scores for plotting)
    print("Running Cross-Validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1
    )
    cv_mean = cv_scores.mean()
    
    # 4. Print Text Results
    print(f"\n=================== {model_name} Results ===================")
    print(" Weighted F1 Score:", round(f1, 4))
    print(" Precision:", round(precision, 4))
    print(" Recall:", round(recall, 4))
    print(" Cross-validation F1 (mean):", round(cv_mean, 4))
    print("\n Classification Report:\n", report)
    
    if f1 >= 0.75:
        print("🎉 Model meets target F1 ≥ 0.75\n")
    else:
        print("❌ Model does NOT meet target F1 = 0.75\n")
        
    # 5. VISUALIZATIONS - Setup Grid Layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Performance Dashboard: {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
    axes[0, 0].set_title("Confusion Matrix", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")
    
    # Plot 2: Core Metrics vs Target Threshold
    metrics_df = pd.DataFrame({
        'Metric': ['Weighted F1', 'Precision', 'Recall', 'CV F1 Mean'],
        'Score': [f1, precision, recall, cv_mean]
    })
    sns.barplot(x='Metric', y='Score', data=metrics_df, palette='viridis', ax=axes[0, 1])
    axes[0, 1].axhline(0.75, color='red', linestyle='--', linewidth=2, label='Target Threshold (0.75)')
    axes[0, 1].set_title("Core Classification Metrics", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()
    
    # Add data labels on top of the bars
    for p in axes[0, 1].patches:
        axes[0, 1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

    # Plot 3: Cross-Validation Fold Stability
    sns.boxplot(y=cv_scores, ax=axes[1, 0], color='#a2d2ff', width=0.4)
    sns.stripplot(y=cv_scores, ax=axes[1, 0], color='black', size=8, jitter=0.1, linewidth=1)
    axes[1, 0].set_title("CV F1 Score Variance Across 5 Folds", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Weighted F1 Score")
    axes[1, 0].axhline(0.75, color='red', linestyle='--', linewidth=1.5)

    # Plot 4: Multi-class ROC Curve (One-vs-Rest)
    if hasattr(model, "predict_proba"):
        from sklearn.preprocessing import LabelBinarizer
        from sklearn.metrics import roc_curve, auc

        # Binarize the labels for multi-class ROC calculation
        lb = LabelBinarizer()
        y_test_binarized = lb.fit_transform(y_test)
        y_prob = model.predict_proba(X_test)
        
        n_classes = y_prob.shape[1]

        if n_classes > 2:
            # Loop through each class and plot its specific One-vs-Rest ROC curve
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                class_label = model.classes_[i] if hasattr(model, "classes_") else i
                axes[1, 1].plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc:.2f})')
            
            axes[1, 1].plot([0, 1], [0, 1], color='navy', linestyle='--')
            axes[1, 1].set_title("Multi-Class ROC Curve (OvR)", fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel("False Positive Rate")
            axes[1, 1].set_ylabel("True Positive Rate")
            axes[1, 1].legend(loc="lower right")
        else:
            # Fallback to standard binary ROC if used for 2 classes
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1, 1], color='darkorange')
            axes[1, 1].plot([0, 1], [0, 1], color='navy', linestyle='--')
            axes[1, 1].set_title("ROC Curve", fontsize=12, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, "ROC Curve unavailable\n(Model doesn't support predict_proba)", 
                        ha='center', va='center', fontsize=12, color='gray')
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "cv_f1": cv_mean
    }