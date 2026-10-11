"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Script: ml/prediction.py

Inference engine: Combines ML model predictions, geospatial blackspot analysis,
environmental factors, and rules-based risk scoring.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

class AccidentRiskPredictor:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.blackspots = []
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(MODELS_DIR, 'accident_risk_model.pkl')
        meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
        blackspots_path = os.path.join(RAW_DIR, 'bengaluru_blackspots.csv')

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        if os.path.exists(blackspots_path):
            df_bs = pd.read_csv(blackspots_path)
            self.blackspots = df_bs.to_dict(orient='records')

    def calculate_haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0 # km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2.0) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon / 2.0) ** 2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def find_nearest_blackspot(self, lat, lon):
        if not self.blackspots:
            return None, 999.0
        
        nearest = None
        min_dist = 999999.0
        for bs in self.blackspots:
            d = self.calculate_haversine(lat, lon, bs['latitude'], bs['longitude'])
            if d < min_dist:
                min_dist = d
                nearest = bs
        return nearest, round(min_dist, 2)

    def predict_location_risk(self, params):
        """
        Predicts accident risk for a single point or corridor.
        Input params dictionary can contain:
        - latitude, longitude
        - area_name (optional, inferred from coords if missing)
        - weather_condition ('Clear', 'Rainy', 'Foggy', 'Overcast', 'Windy')
        - traffic_volume (default estimated if missing)
        - average_speed (default estimated)
        - congestion_level (0-100)
        - roadwork_present (0 or 1)
        - date (optional, defaults to now)
        """
        now = datetime.now()
        lat = float(params.get('latitude', 12.9716))
        lon = float(params.get('longitude', 77.5946))
        
        nearest_bs, bs_dist = self.find_nearest_blackspot(lat, lon)
        
        area_name = params.get('area_name', 'Bengaluru Central')
        weather = params.get('weather_condition', 'Clear')
        congestion = float(params.get('congestion_level', 65.0))
        traffic_vol = float(params.get('traffic_volume', 32000.0))
        speed = float(params.get('average_speed', 35.0))
        roadwork = int(params.get('roadwork_present', 0))
        pedestrians = int(params.get('pedestrian_cyclist_count', 90))
        
        month = int(params.get('month', now.month))
        day_of_week = int(params.get('day_of_week', now.weekday()))
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # Create input dataframe matching model features
        input_data = pd.DataFrame([{
            'traffic_volume': traffic_vol,
            'average_speed': speed,
            'travel_time_index': round(np.clip(congestion / 50.0, 1.0, 3.0), 2),
            'congestion_level': congestion,
            'road_capacity_utilization': np.clip(congestion + 5.0, 10.0, 100.0),
            'pedestrian_cyclist_count': pedestrians,
            'roadwork_present': roadwork,
            'distance_to_nearest_blackspot_km': bs_dist,
            'latitude': lat,
            'longitude': lon,
            'month': month,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'weather_condition': weather,
            'area_name': area_name
        }])

        if self.model is not None:
            pred_code = int(self.model.predict(input_data)[0])
            probas = self.model.predict_proba(input_data)[0]
            confidence = float(probas[pred_code])
        else:
            # Fallback heuristic if model file is reloading
            pred_code = 2 if (bs_dist < 0.5 or congestion > 85 or weather == 'Rainy') else (1 if congestion > 50 else 0)
            probas = [0.2, 0.4, 0.4]
            confidence = 0.5

        classes = ['Low Risk', 'Medium Risk', 'High Risk']
        pred_label = classes[pred_code]

        # Calculate composite 0-100 risk score
        base_score = 30 if pred_code == 0 else (60 if pred_code == 1 else 85)
        # Factor adjustments
        weather_adj = 12 if weather in ['Rainy', 'Foggy'] else (4 if weather == 'Overcast' else 0)
        congestion_adj = (congestion / 100.0) * 15
        blackspot_adj = 15 if bs_dist < 0.4 else (8 if bs_dist < 1.0 else 0)
        roadwork_adj = 8 if roadwork == 1 else 0

        composite_score = int(np.clip(base_score * 0.5 + weather_adj + congestion_adj + blackspot_adj + roadwork_adj, 10, 99))

        # Identify Primary Risk Factors
        risk_factors = []
        if bs_dist <= 0.8 and nearest_bs:
            risk_factors.append(f"Proximity to BTP Blackspot: '{nearest_bs['blackspot_name']}' ({bs_dist * 1000:.0f}m away)")
        if weather in ['Rainy', 'Foggy']:
            risk_factors.append(f"Adverse weather condition detected: {weather} (reduced traction / visibility)")
        if congestion >= 80:
            risk_factors.append(f"Severe corridor congestion ({congestion:.0f}%), leading to stop-and-go collision risk")
        if speed > 60:
            risk_factors.append(f"High vehicular speed ({speed:.0f} km/h) on arterial corridor")
        elif speed < 15 and congestion > 85:
            risk_factors.append("Severe gridlock friction with heavy bumper-to-bumper collision risk")
        if roadwork == 1:
            risk_factors.append("Active construction or roadwork narrowing available carriageway")
        if pedestrians > 120:
            risk_factors.append(f"High vulnerable road user activity ({pedestrians} pedestrians/cyclists)")
        if is_weekend and now.hour >= 20:
            risk_factors.append("Weekend night travel: elevated risk of speeding and impaired driving")

        if not risk_factors:
            risk_factors.append("Normal traffic flow with standard urban driving hazards")

        # Actionable Safety Recommendations
        recommendations = self._generate_recommendations(pred_code, weather, roadwork, bs_dist, congestion)

        return {
            'predicted_risk_level': pred_label,
            'risk_code': pred_code,
            'confidence_score': round(confidence * 100, 1),
            'composite_risk_score': composite_score,
            'class_probabilities': {
                'Low Risk': round(float(probas[0]) * 100, 1),
                'Medium Risk': round(float(probas[1]) * 100, 1),
                'High Risk': round(float(probas[2]) * 100, 1)
            },
            'nearest_blackspot': {
                'name': nearest_bs['blackspot_name'] if nearest_bs else 'None nearby',
                'distance_km': bs_dist,
                'hazard_factors': nearest_bs.get('primary_hazard_factors', 'N/A') if nearest_bs else 'None',
                'historical_accidents': nearest_bs.get('historical_3yr_accidents', 0) if nearest_bs else 0
            },
            'primary_risk_factors': risk_factors,
            'safety_recommendations': recommendations
        }

    def _generate_recommendations(self, risk_code, weather, roadwork, bs_dist, congestion):
        recs = []
        if weather == 'Rainy':
            recs.append("Wet asphalt increases braking distance by 40% - maintain at least 3-car gap")
            recs.append("Turn on low-beam headlights and ensure wiper blades operate at adequate speed")
            recs.append("Avoid sudden lane swerves across waterlogged shoulder lanes")
        elif weather == 'Foggy':
            recs.append("Visibility compromised - turn on front and rear fog lights and avoid overtaking")

        if bs_dist < 0.5:
            recs.append("Entering verified high-fatality blackspot junction - obey speed limits strictly and watch for merging traffic")

        if roadwork == 1:
            recs.append("Active roadwork zone - reduce speed to 25 km/h and watch for construction machinery and barricades")

        if congestion > 80:
            recs.append("Heavy bottleneck - beware of sudden tail-end stops and aggressive two-wheeler filtering")

        # General safety advice based on predicted tier
        if risk_code == 2: # High Risk
            recs.append("HIGH RISK ALERT: Avoid high-speed acceleration; consider alternative bypass routes if available")
            recs.append("Two-wheeler riders must wear ISI-certified helmets with visors fastened securely")
        elif risk_code == 1: # Medium Risk
            recs.append("Drive with elevated vigilance at upcoming junctions and pedestrian zebra crossings")
        else: # Low Risk
            recs.append("Conditions are favorable - maintain standard city cruising speed and observe traffic signals")

        return recs

_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AccidentRiskPredictor()
    return _predictor_instance
