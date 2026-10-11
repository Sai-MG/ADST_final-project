"""
Unit tests for Dataset loading, columns, and preprocessing integrity.
"""

import os
import unittest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'bengaluru_accidents_clean.csv')
BLACKSPOTS_CSV = os.path.join(BASE_DIR, 'data', 'raw', 'bengaluru_blackspots.csv')
REPORT_JSON = os.path.join(BASE_DIR, 'data', 'processed', 'preprocessing_report.json')

class TestDataPipeline(unittest.TestCase):
    def test_cleaned_dataset_exists(self):
        self.assertTrue(os.path.exists(PROCESSED_CSV), "Cleaned dataset CSV does not exist.")

    def test_cleaned_dataset_rows_and_cols(self):
        df = pd.read_csv(PROCESSED_CSV)
        self.assertGreaterEqual(len(df), 8000, "Dataset should have at least 8000 rows.")
        self.assertIn('risk_level', df.columns, "Missing risk_level target.")
        self.assertIn('weather_condition', df.columns, "Missing weather_condition.")
        self.assertIn('latitude', df.columns, "Missing latitude.")
        self.assertIn('longitude', df.columns, "Missing longitude.")
        self.assertIn('traffic_volume', df.columns, "Missing traffic_volume.")

    def test_no_critical_nulls(self):
        df = pd.read_csv(PROCESSED_CSV)
        critical_cols = ['risk_level', 'latitude', 'longitude', 'traffic_volume', 'weather_condition']
        for col in critical_cols:
            self.assertEqual(df[col].isnull().sum(), 0, f"Column {col} contains unexpected null values.")

    def test_official_blackspots_file(self):
        self.assertTrue(os.path.exists(BLACKSPOTS_CSV), "Blackspots reference CSV not found.")
        df_bs = pd.read_csv(BLACKSPOTS_CSV)
        self.assertGreaterEqual(len(df_bs), 20, "Should have at least 20 official blackspots.")
        self.assertIn('blackspot_name', df_bs.columns)
        self.assertIn('historical_3yr_accidents', df_bs.columns)

    def test_preprocessing_report_generated(self):
        self.assertTrue(os.path.exists(REPORT_JSON), "Preprocessing report JSON not found.")

if __name__ == '__main__':
    unittest.main()
