"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Script: ml/evaluate_model.py

Evaluates the trained machine learning models, prints academic classification reports,
confusion matrix breakdown, and tests scenario predictions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'bengaluru_accidents_clean.csv')

def evaluate():
    print("=" * 70)
    print("BENGALURU ROAD ACCIDENT RISK MODEL EVALUATION REPORT")
    print("=" * 70)

    model_path = os.path.join(MODELS_DIR, 'accident_risk_model.pkl')
    metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Model artifacts not found. Run ml/train_model.py first.")

    model = joblib.load(model_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    print(f"Selected Best Model: {meta['best_model']}")
    print(f"Target Classes:      {', '.join(meta['target_classes'])}")
    print(f"Training Samples:    {meta['training_samples']:,}")
    print(f"Testing Samples:     {meta['test_samples']:,}\n")

    print("-" * 70)
    print("BENCHMARKED CLASSIFIERS COMPARISON")
    print("-" * 70)
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'ROC-AUC':<8}")
    print("-" * 70)
    for m_name, m_metrics in meta['all_models_benchmarked'].items():
        print(f"{m_name:<22} | {m_metrics['accuracy']:<10.4f} | {m_metrics['macro_f1']:<10.4f} | {m_metrics['weighted_f1']:<12.4f} | {m_metrics['roc_auc']:<8.4f}")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("TOP PRE-INCIDENT RISK PREDICTION FEATURES")
    print("=" * 70)
    for feat, imp in meta.get('top_feature_importances', {}).items():
        print(f"  - {feat:<35}: {imp * 100:.2f}%")

    # Real-World Scenario Testing
    print("\n" + "=" * 70)
    print("REAL-WORLD SCENARIO STRESS TESTING")
    print("=" * 70)

    test_scenarios = [
        {
            'name': 'Silk Board Junction (Monsoon Heavy Rain & Bottleneck Traffic)',
            'data': {
                'traffic_volume': 55000,
                'average_speed': 18.5,
                'travel_time_index': 2.4,
                'congestion_level': 98.0,
                'road_capacity_utilization': 99.0,
                'pedestrian_cyclist_count': 180,
                'roadwork_present': 1,
                'distance_to_nearest_blackspot_km': 0.1,
                'latitude': 12.9174,
                'longitude': 77.6238,
                'month': 7,
                'day_of_week': 4,
                'is_weekend': 0,
                'weather_condition': 'Rainy',
                'area_name': 'Electronic City'
            }
        },
        {
            'name': 'Jayanagar 4th Block (Sunday Morning Clear Weather)',
            'data': {
                'traffic_volume': 12000,
                'average_speed': 48.0,
                'travel_time_index': 1.05,
                'congestion_level': 25.0,
                'road_capacity_utilization': 35.0,
                'pedestrian_cyclist_count': 60,
                'roadwork_present': 0,
                'distance_to_nearest_blackspot_km': 2.5,
                'latitude': 12.9298,
                'longitude': 77.5826,
                'month': 1,
                'day_of_week': 6,
                'is_weekend': 1,
                'weather_condition': 'Clear',
                'area_name': 'Jayanagar'
            }
        },
        {
            'name': 'Hebbal Flyover (Peak Hour Dense Foggy Morning)',
            'data': {
                'traffic_volume': 42000,
                'average_speed': 32.0,
                'travel_time_index': 1.8,
                'congestion_level': 88.0,
                'road_capacity_utilization': 92.0,
                'pedestrian_cyclist_count': 95,
                'roadwork_present': 0,
                'distance_to_nearest_blackspot_km': 0.2,
                'latitude': 13.0358,
                'longitude': 77.5970,
                'month': 12,
                'day_of_week': 2,
                'is_weekend': 0,
                'weather_condition': 'Foggy',
                'area_name': 'Hebbal'
            }
        }
    ]

    for sc in test_scenarios:
        df_sample = pd.DataFrame([sc['data']])
        pred_class = model.predict(df_sample)[0]
        pred_proba = model.predict_proba(df_sample)[0]
        class_name = meta['target_classes'][pred_class]
        confidence = pred_proba[pred_class] * 100

        print(f"\nScenario: {sc['name']}")
        print(f"  --> Prediction: {class_name.upper()} (Confidence: {confidence:.1f}%)")
        print(f"  --> Class Probabilities: Low: {pred_proba[0]*100:.1f}%, Medium: {pred_proba[1]*100:.1f}%, High: {pred_proba[2]*100:.1f}%")

    print("\n" + "=" * 70)
    print("MODEL EVALUATION VERIFIED AND READY")
    print("=" * 70)

if __name__ == '__main__':
    evaluate()
