"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Module: api/maps.py

Geospatial data providers and route risk evaluation engine for Leaflet maps.
"""

import os
import pandas as pd
import numpy as np
from ml.prediction import get_predictor
from api.traffic import geocode_location, get_route
from api.weather import get_live_weather

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def get_blackspots_geojson():
    """Returns official Bengaluru accident blackspots as GeoJSON FeatureCollection."""
    blackspots_path = os.path.join(RAW_DIR, 'bengaluru_blackspots.csv')
    features = []
    if os.path.exists(blackspots_path):
        df = pd.read_csv(blackspots_path)
        for _, r in df.iterrows():
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [float(r['longitude']), float(r['latitude'])]
                },
                'properties': {
                    'name': str(r['blackspot_name']),
                    'corridor': str(r['corridor']),
                    'station': str(r['btp_police_station']),
                    'risk_category': str(r['risk_category']),
                    'accidents_3yr': int(r['historical_3yr_accidents']),
                    'hazards': str(r['primary_hazard_factors'])
                }
            })
    return {
        'type': 'FeatureCollection',
        'features': features
    }

def get_btp_stations_geojson():
    """Returns official BTP Traffic Police Stations and casualty stats as GeoJSON."""
    summary_path = os.path.join(PROCESSED_DIR, 'btp_station_risk_summary.csv')
    features = []
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        for _, r in df.iterrows():
            lat = float(r.get('latitude', 12.9716))
            lon = float(r.get('longitude', 77.5946))
            station_name = str(r.get('Station', 'Unknown'))
            total = int(r.get('2024-Total crashes', 0))
            fatal = int(r.get('2024-Fatal crashes', 0))
            non_fatal = int(r.get('2024-Non Fatal', 0))

            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lon, lat]
                },
                'properties': {
                    'station': station_name,
                    'total_crashes': total,
                    'fatal_crashes': fatal,
                    'non_fatal_crashes': non_fatal,
                    'zone': str(r.get('Zone', 'Central'))
                }
            })
    return {
        'type': 'FeatureCollection',
        'features': features
    }

def evaluate_route_risk(origin_query, destination_query):
    """
    Evaluates end-to-end accident risk along a route between origin and destination.
    1. Geocodes origin & destination
    2. Fetches real driving route waypoints via OSRM
    3. Evaluates historical accident risk at sampled route segments
    4. Gathers real-time live weather
    5. Checks live/historical traffic congestion
    6. Produces overall Route Risk Score (0-100), risk tier, and key risk factors
    """
    orig_lat, orig_lon, orig_name = geocode_location(origin_query)
    dest_lat, dest_lon, dest_name = geocode_location(destination_query)

    route_info = get_route(orig_lat, orig_lon, dest_lat, dest_lon)
    weather_info = get_live_weather(lat=orig_lat, lon=orig_lon)

    predictor = get_predictor()
    waypoints = route_info.get('waypoints', [])

    # Sample ~10-15 key checkpoints along the route
    sample_step = max(1, len(waypoints) // 12)
    sampled_points = waypoints[::sample_step]
    if waypoints and waypoints[-1] not in sampled_points:
        sampled_points.append(waypoints[-1])

    checkpoint_predictions = []
    encountered_blackspots = []
    point_scores = []

    for idx, (lat, lon) in enumerate(sampled_points):
        # Determine nearest blackspot
        nearest_bs, bs_dist = predictor.find_nearest_blackspot(lat, lon)
        if bs_dist <= 0.6 and nearest_bs:
            bs_name = nearest_bs['blackspot_name']
            if bs_name not in [b['name'] for b in encountered_blackspots]:
                encountered_blackspots.append({
                    'name': bs_name,
                    'distance_m': int(bs_dist * 1000),
                    'risk_category': nearest_bs.get('risk_category', 'HIGH'),
                    'hazard_factors': nearest_bs.get('primary_hazard_factors', ''),
                    'historical_accidents': nearest_bs.get('historical_3yr_accidents', 0)
                })

        # Run ML model on checkpoint
        chk_res = predictor.predict_location_risk({
            'latitude': lat,
            'longitude': lon,
            'area_name': 'Route Segment',
            'weather_condition': weather_info['condition'],
            'congestion_level': 85.0 if route_info['traffic_status'] in ['Evening Peak Gridlock', 'Morning Peak Rush', 'Heavy Traffic'] else 50.0,
            'average_speed': 25.0 if route_info['traffic_status'] in ['Evening Peak Gridlock', 'Morning Peak Rush', 'Heavy Traffic'] else 42.0
        })

        checkpoint_predictions.append({
            'lat': lat,
            'lon': lon,
            'checkpoint_id': idx + 1,
            'risk_level': chk_res['predicted_risk_level'],
            'risk_score': chk_res['composite_risk_score']
        })
        point_scores.append(chk_res['composite_risk_score'])

    # Aggregate overall route risk
    avg_score = float(np.mean(point_scores)) if point_scores else 50.0
    max_score = float(np.max(point_scores)) if point_scores else 50.0

    # Weight route risk: 60% average along journey + 40% peak hazard point
    composite_route_score = int(np.clip(0.6 * avg_score + 0.4 * max_score, 10, 99))

    if composite_route_score >= 70:
        overall_risk_tier = 'HIGH RISK'
        risk_color = '#e74c3c' # Red
    elif composite_route_score >= 45:
        overall_risk_tier = 'MEDIUM RISK'
        risk_color = '#f39c12' # Orange/Yellow
    else:
        overall_risk_tier = 'LOW RISK'
        risk_color = '#2ecc71' # Green

    # Compile Main Risk Factors
    main_risk_factors = []
    if encountered_blackspots:
        bs_names = ", ".join([f"'{b['name']}'" for b in encountered_blackspots[:3]])
        main_risk_factors.append(f"Route intersects or passes in close proximity to {len(encountered_blackspots)} major accident blackspot(s): {bs_names}.")
    
    if weather_info['condition'] == 'Rainy':
        main_risk_factors.append(f"Adverse weather condition active: Heavy rain ({weather_info['rain_mm']} mm/hr) causing wet asphalt and reduced braking grip.")
    elif weather_info['condition'] == 'Foggy':
        main_risk_factors.append(f"Poor visibility ({weather_info['visibility_km']} km) due to fog/mist.")

    if route_info['delay_min'] > 10 or route_info['traffic_status'] in ['Evening Peak Gridlock', 'Morning Peak Rush', 'Heavy Traffic']:
        main_risk_factors.append(f"Severe corridor traffic congestion detected: {route_info['traffic_status']} (estimated delay: +{route_info['delay_min']} mins).")

    if route_info['distance_km'] > 18:
        main_risk_factors.append(f"Long transit corridor ({route_info['distance_km']} km) spanning multiple dense arterial intersections.")

    if not main_risk_factors:
        main_risk_factors.append("Moderate urban traffic conditions with standard city driving hazards.")

    return {
        'origin': {
            'query': origin_query,
            'name': orig_name,
            'lat': orig_lat,
            'lon': orig_lon
        },
        'destination': {
            'query': destination_query,
            'name': dest_name,
            'lat': dest_lat,
            'lon': dest_lon
        },
        'route': {
            'distance_km': route_info['distance_km'],
            'duration_min': route_info['duration_min'],
            'delay_min': route_info['delay_min'],
            'traffic_status': route_info['traffic_status'],
            'traffic_note': route_info['traffic_note'],
            'is_live_traffic': route_info['is_live_traffic'],
            'waypoints': waypoints
        },
        'weather': weather_info,
        'overall_risk': {
            'risk_tier': overall_risk_tier,
            'risk_score': composite_route_score,
            'risk_color': risk_color,
            'encountered_blackspots_count': len(encountered_blackspots),
            'encountered_blackspots': encountered_blackspots,
            'main_risk_factors': main_risk_factors
        },
        'checkpoint_samples': checkpoint_predictions
    }
