"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Script: ml/train_model.py

Genuine Machine Learning Pipeline:
- Strictly prevents Data Leakage (only pre-incident factors used)
- Evaluates multiple classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Addresses class imbalance with balanced weighting
- Compares models via Accuracy, Precision, Recall, Macro F1, Weighted F1, ROC-AUC
- Selects the optimal model and exports model artifacts + metadata
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'bengaluru_accidents_clean.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Risk class labels
TARGET_CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

def train_and_evaluate():
    print("=" * 70)
    print("BENGALURU ROAD ACCIDENT RISK MODEL TRAINING & EVALUATION")
    print("=" * 70)

    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(f"Missing {PROCESSED_CSV}. Run data/preprocess.py first.")

    df = pd.read_csv(PROCESSED_CSV)
    print(f"Loaded dataset: {len(df):,} records")

    # Define Pre-Incident Prediction Features (STRICTLY NO DATA LEAKAGE)
    numeric_features = [
        'traffic_volume',
        'average_speed',
        'travel_time_index',
        'congestion_level',
        'road_capacity_utilization',
        'pedestrian_cyclist_count',
        'roadwork_present',
        'distance_to_nearest_blackspot_km',
        'latitude',
        'longitude',
        'month',
        'day_of_week',
        'is_weekend'
    ]

    categorical_features = [
        'weather_condition',
        'area_name'
    ]

    # Target variable: multi-class accident risk level (0: Low, 1: Medium, 2: High)
    target_col = 'risk_level'

    X = df[numeric_features + categorical_features].copy()
    y = df[target_col].copy()

    print(f"\nFeature Matrix X shape: {X.shape}")
    print(f"Class distribution: {dict(y.value_counts().sort_index())}")
    for idx, name in enumerate(TARGET_CLASSES):
        cnt = (y == idx).sum()
        print(f"  Class {idx} ({name}): {cnt:,} samples ({cnt/len(y)*100:.1f}%)")

    # Train / Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # Preprocessing Pipeline (Scaler + OneHotEncoder)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Models to benchmark
    candidate_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }

    results = {}
    fitted_pipelines = {}

    print("\n" + "-" * 70)
    print(f"{'Model Name':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'Macro F1':<8} | {'ROC-AUC':<8}")
    print("-" * 70)

    for name, clf in candidate_models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Train pipeline
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Predictions
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0

        results[name] = {
            'accuracy': float(acc),
            'precision_weighted': float(prec),
            'recall_weighted': float(rec),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'roc_auc': float(auc),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, target_names=TARGET_CLASSES, output_dict=True)
        }

        print(f"{name:<22} | {acc:.4f}   | {prec:.4f}    | {rec:.4f}   | {macro_f1:.4f}   | {auc:.4f}")

    print("-" * 70)

    # Model Selection: select best model based on Macro F1 (balances all risk tiers)
    best_model_name = max(results.keys(), key=lambda k: results[k]['macro_f1'])
    best_pipe = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\n[SELECTION] Best Performing Model: >>> {best_model_name} <<<")
    print(f"  Macro F1-Score: {best_metrics['macro_f1']:.4f}")
    print(f"  Accuracy:       {best_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:        {best_metrics['roc_auc']:.4f}")

    # Save Model Artifacts
    model_path = os.path.join(MODELS_DIR, 'accident_risk_model.pkl')
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    
    # Save the complete pipeline as well as standalone objects
    joblib.dump(best_pipe, model_path)
    joblib.dump(best_pipe.named_steps['preprocessor'], preprocessor_path)
    print(f"\n[SAVED] Best Model Pipeline -> {model_path}")
    print(f"[SAVED] Preprocessing Transformer -> {preprocessor_path}")

    # Feature importances if available
    feature_importances = {}
    try:
        clf_step = best_pipe.named_steps['classifier']
        if hasattr(clf_step, 'feature_importances_'):
            prep_step = best_pipe.named_steps['preprocessor']
            cat_encoder = prep_step.named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
            all_feature_names = numeric_features + cat_feature_names
            
            imps = clf_step.feature_importances_
            feature_importances = {
                name: float(imp) for name, imp in sorted(zip(all_feature_names, imps), key=lambda x: x[1], reverse=True)[:15]
            }
    except Exception as e:
        print(f"[NOTE] Feature importances extraction note: {e}")

    # Save Metadata JSON
    metadata = {
        'best_model': best_model_name,
        'target_classes': TARGET_CLASSES,
        'metrics': best_metrics,
        'all_models_benchmarked': {k: {
            'accuracy': v['accuracy'],
            'macro_f1': v['macro_f1'],
            'weighted_f1': v['weighted_f1'],
            'roc_auc': v['roc_auc']
        } for k, v in results.items()},
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'top_feature_importances': feature_importances,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'data_source': 'Bengaluru Real-World Traffic & Accident Corridor Dataset (2022-2023)'
    }

    metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"[SAVED] Model Metadata -> {metadata_path}")

    # Visualizations: Model Comparison & Confusion Matrix
    save_evaluation_plots(results, best_model_name, y_test, best_pipe.predict(X_test))

    print("=" * 70)
    print("TRAINING PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)
    return best_pipe, metadata

def save_evaluation_plots(results, best_model_name, y_test, y_pred):
    """Generates and saves model comparison bar chart and best model confusion matrix."""
    # 1. Model Comparison Chart
    plt.figure(figsize=(10, 5))
    model_names = list(results.keys())
    macro_f1s = [results[m]['macro_f1'] for m in model_names]
    accuracies = [results[m]['accuracy'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(x - width/2, macro_f1s, width, label='Macro F1-Score', color='#3498db', edgecolor='black')
    plt.bar(x + width/2, accuracies, width, label='Accuracy', color='#2ecc71', edgecolor='black')

    plt.xlabel('Machine Learning Algorithm', fontsize=12)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.title('Performance Comparison of Accident Risk Classifiers', fontsize=14, fontweight='bold', pad=12)
    plt.xticks(x, model_names, fontsize=11)
    plt.ylim(0, 1.1)
    plt.legend(frameon=True)

    for i in range(len(model_names)):
        plt.text(i - width/2, macro_f1s[i] + 0.02, f"{macro_f1s[i]:.2f}", ha='center', fontsize=9, fontweight='bold')
        plt.text(i + width/2, accuracies[i] + 0.02, f"{accuracies[i]:.2f}", ha='center', fontsize=9, fontweight='bold')

    out1 = os.path.join(FIGURES_DIR, 'model_comparison.png')
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"[SAVED] {out1}")

    # 2. Confusion Matrix for Best Model
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=TARGET_CLASSES,
        yticklabels=TARGET_CLASSES,
        cbar=False
    )
    plt.title(f'Confusion Matrix: {best_model_name}', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Risk Class', fontsize=11)
    plt.ylabel('Actual Historical Risk Class', fontsize=11)
    out2 = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"[SAVED] {out2}")

if __name__ == '__main__':
    train_and_evaluate()
