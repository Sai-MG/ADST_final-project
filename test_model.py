"""
Unit tests for Machine Learning model prediction, metadata, and probabilities.
"""

import os
import unittest
from ml.prediction import get_predictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'accident_risk_model.pkl')
META_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

class TestMLModel(unittest.TestCase):
    def setUp(self):
        self.predictor = get_predictor()

    def test_model_artifact_exists(self):
        self.assertTrue(os.path.exists(MODEL_PATH), "Trained model pkl not found.")
        self.assertTrue(os.path.exists(META_PATH), "Model metadata json not found.")

    def test_prediction_output_structure(self):
        res = self.predictor.predict_location_risk({
            'latitude': 12.9174,
            'longitude': 77.6238,
            'area_name': 'Electronic City',
            'weather_condition': 'Rainy',
            'congestion_level': 90.0
        })

        self.assertIn('predicted_risk_level', res)
        self.assertIn('composite_risk_score', res)
        self.assertIn('class_probabilities', res)
        self.assertIn('nearest_blackspot', res)
        self.assertIn('primary_risk_factors', res)
        self.assertIn('safety_recommendations', res)

        # Probabilities should sum to approximately 100%
        probas = res['class_probabilities']
        total_prob = sum(probas.values())
        self.assertAlmostEqual(total_prob, 100.0, delta=1.5)

    def test_high_risk_scenario(self):
        # Peak congestion, adverse rain near Silk Board
        res = self.predictor.predict_location_risk({
            'latitude': 12.9174,
            'longitude': 77.6238,
            'area_name': 'Electronic City',
            'weather_condition': 'Rainy',
            'congestion_level': 98.0,
            'roadwork_present': 1
        })
        self.assertGreaterEqual(res['composite_risk_score'], 60, "Severe scenario should yield high score.")

    def test_low_risk_scenario(self):
        # Sunday morning clear conditions away from blackspots
        res = self.predictor.predict_location_risk({
            'latitude': 12.9300,
            'longitude': 77.5800,
            'area_name': 'Jayanagar',
            'weather_condition': 'Clear',
            'congestion_level': 20.0,
            'day_of_week': 6,
            'is_weekend': 1
        })
        self.assertLessEqual(res['composite_risk_score'], 70, "Mild conditions should score lower.")

if __name__ == '__main__':
    unittest.main()
