"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Module: api/traffic.py

Legitimate Route & Traffic API Integration:
- Geocoding: Bengaluru Landmark Catalog + OpenStreetMap Nominatim
- Routing: Open Source Routing Machine (OSRM) driving route geometry & distance
- Live Traffic Delay: TomTom Traffic Flow / Mapbox (if TRAFFIC_API_KEY configured)
- Graceful Fallback: Historical Corridor Traffic baseline with explicit transparency message
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

TRAFFIC_API_KEY = os.getenv('TRAFFIC_API_KEY', '').strip()

# Curated reference coordinates for prominent Bengaluru locations
BENGALURU_LANDMARKS = {
    'majestic': (12.9767, 77.5713, 'Majestic (Kempegowda Bus Station)'),
    'kempegowda bus station': (12.9767, 77.5713, 'Majestic Bus Station'),
    'silk board': (12.9174, 77.6238, 'Silk Board Junction (Hosur Road / ORR)'),
    'central silk board': (12.9174, 77.6238, 'Central Silk Board'),
    'hebbal': (13.0358, 77.5970, 'Hebbal Flyover (Airport Corridor)'),
    'hebbal flyover': (13.0358, 77.5970, 'Hebbal Flyover'),
    'whitefield': (12.9698, 77.7499, 'Whitefield (ITPL)'),
    'itpl': (12.9856, 77.7315, 'ITPL Main Road Whitefield'),
    'indiranagar': (12.9719, 77.6412, 'Indiranagar 100 Feet Road'),
    'koramangala': (12.9345, 77.6266, 'Koramangala (Sony World Junction)'),
    'electronic city': (12.8452, 77.6602, 'Electronic City Phase 1 Tollgate'),
    'jayanagar': (12.9298, 77.5826, 'Jayanagar 4th Block'),
    'marathahalli': (12.9562, 77.7011, 'Marathahalli Bridge'),
    'kr puram': (13.0031, 77.6893, 'K.R. Puram Hanging Bridge'),
    'tin factory': (12.9937, 77.6606, 'Tin Factory Junction (Old Madras Rd)'),
    'yeshwanthpur': (13.0213, 77.5501, 'Yeshwanthpur Circle'),
    'mg road': (12.9734, 77.6205, 'M.G. Road (Trinity Circle)'),
    'banashankari': (12.9255, 77.5736, 'Banashankari Temple Junction'),
    'hsr layout': (12.9121, 77.6446, 'HSR Layout BDA Complex'),
    'bellandur': (12.9260, 77.6762, 'Bellandur EcoSpace (Outer Ring Road)'),
    'mysore road': (12.9535, 77.5412, 'Mysore Road Tollgate'),
    'kengeri': (12.9116, 77.4831, 'Kengeri Satellite Bus Stand'),
    'yelahanka': (13.1007, 77.5963, 'Yelahanka NES Circle'),
    'peenya': (13.0298, 77.5186, 'Peenya 1st Stage Industrial Area'),
    'goraguntepalya': (13.0285, 77.5408, 'Goraguntepalya Junction')
}

def geocode_location(query):
    """
    Geocodes a Bengaluru location string to (latitude, longitude, display_name).
    Uses fast local dictionary first, then falls back to OpenStreetMap Nominatim.
    """
    q_norm = query.strip().lower()
    for key, (lat, lon, name) in BENGALURU_LANDMARKS.items():
        if key in q_norm or q_norm in key:
            return lat, lon, name

    # Fallback to OpenStreetMap Nominatim geocoding (restricted to Bengaluru)
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': f"{query}, Bengaluru, Karnataka, India",
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'BengaluruRoadSafetyDSApp/1.0'}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            results = resp.json()
            if results:
                lat = float(results[0]['lat'])
                lon = float(results[0]['lon'])
                name = results[0].get('display_name', query)
                return lat, lon, name
    except Exception as e:
        print(f"[WARN] Geocoding lookup error: {e}")

    # Central Bengaluru default fallback
    return 12.9716, 77.5946, f"{query} (Bengaluru approx)"

def get_route(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    Computes road routing between origin and destination.
    Uses OSRM (Open Source Routing Machine) public driving API.
    Returns:
    - distance_km
    - duration_min
    - route_coordinates: list of [lat, lon] waypoints
    - is_live_traffic: bool
    - traffic_note: str
    """
    # 1. Fetch real geometry from OSRM
    osrm_url = f"https://router.project-osrm.org/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=full&geometries=geojson"
    
    try:
        resp = requests.get(osrm_url, headers={'User-Agent': 'BengaluruAccidentSystem/1.0'}, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('routes'):
                route = data['routes'][0]
                dist_meters = route.get('distance', 0)
                duration_sec = route.get('duration', 0)
                # GeoJSON coordinates are [lon, lat], convert to [lat, lon] for Leaflet
                coords_geojson = route.get('geometry', {}).get('coordinates', [])
                waypoints = [[pt[1], pt[0]] for pt in coords_geojson]
                
                dist_km = round(dist_meters / 1000.0, 2)
                base_duration_min = round(duration_sec / 60.0, 1)

                # Check if TomTom API is configured for real-time traffic
                traffic_info = check_live_traffic_api(origin_lat, origin_lon, dest_lat, dest_lon)

                if traffic_info['is_live']:
                    est_duration_min = traffic_info['duration_min']
                    traffic_status = traffic_info['traffic_status']
                    delay_min = traffic_info['delay_min']
                    traffic_note = traffic_info['note']
                else:
                    # Realistic peak/off-peak congestion factor based on time of day
                    est_duration_min, traffic_status, delay_min, traffic_note = estimate_historical_traffic_delay(base_duration_min)

                return {
                    'status': 'SUCCESS',
                    'distance_km': dist_km,
                    'base_duration_min': base_duration_min,
                    'duration_min': est_duration_min,
                    'delay_min': delay_min,
                    'traffic_status': traffic_status,
                    'traffic_note': traffic_note,
                    'is_live_traffic': traffic_info['is_live'],
                    'waypoints': waypoints,
                    'waypoints_count': len(waypoints)
                }
    except Exception as e:
        print(f"[WARN] OSRM routing request failed: {e}")

    # Fallback straight-line / Euclidean path if OSRM is unreachable
    dlat = dest_lat - origin_lat
    dlon = dest_lon - origin_lon
    steps = 15
    waypoints = [
        [origin_lat + dlat * (i / steps), origin_lon + dlon * (i / steps)]
        for i in range(steps + 1)
    ]
    # Haversine distance
    from ml.prediction import get_predictor
    p = get_predictor()
    dist_km = round(p.calculate_haversine(origin_lat, origin_lon, dest_lat, dest_lon) * 1.3, 2)
    dur_min = round(dist_km / 22.0 * 60, 1) # ~22 km/h avg city speed

    return {
        'status': 'SUCCESS_FALLBACK',
        'distance_km': dist_km,
        'base_duration_min': dur_min,
        'duration_min': dur_min,
        'delay_min': 0.0,
        'traffic_status': 'Historical Average',
        'traffic_note': 'Live routing service currently unavailable. Approximated via corridor geometry.',
        'is_live_traffic': False,
        'waypoints': waypoints,
        'waypoints_count': len(waypoints)
    }

def check_live_traffic_api(lat1, lon1, lat2, lon2):
    """
    Checks if TomTom or other configured provider has a valid key.
    If not, transparently signals that live traffic API is unavailable.
    """
    if not TRAFFIC_API_KEY:
        return {
            'is_live': False,
            'duration_min': 0,
            'delay_min': 0,
            'traffic_status': 'Live traffic data unavailable',
            'note': 'Live traffic service requires TRAFFIC_API_KEY. Using empirical historical traffic congestion models.'
        }

    try:
        # TomTom Routing API
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json?key={TRAFFIC_API_KEY}&traffic=true"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            summary = data['routes'][0]['summary']
            delay_sec = summary.get('trafficDelayInSeconds', 0)
            travel_sec = summary.get('travelTimeInSeconds', 0)
            return {
                'is_live': True,
                'duration_min': round(travel_sec / 60.0, 1),
                'delay_min': round(delay_sec / 60.0, 1),
                'traffic_status': 'Heavy Traffic' if delay_sec > 600 else ('Moderate' if delay_sec > 180 else 'Normal Flow'),
                'note': f"Live traffic telemetry from TomTom API (Traffic delay: {delay_sec // 60} mins)."
            }
    except Exception as e:
        print(f"[WARN] Live traffic API check failed: {e}")

    return {
        'is_live': False,
        'duration_min': 0,
        'delay_min': 0,
        'traffic_status': 'Live traffic data unavailable',
        'note': 'Live traffic API temporarily unreachable. Historical Bengaluru congestion patterns applied.'
    }

def estimate_historical_traffic_delay(base_duration_min):
    """
    Estimates congestion based on Bengaluru historical peak hours:
    Morning Peak: 8:30 AM - 11:30 AM (factor 1.65)
    Evening Peak: 5:30 PM - 9:30 PM (factor 1.85)
    Off-Peak: 11:30 AM - 5:00 PM (factor 1.30)
    Night: 10:00 PM - 6:00 AM (factor 1.05)
    """
    from datetime import datetime
    now = datetime.now()
    hour = now.hour + now.minute / 60.0

    if (8.5 <= hour <= 11.5):
        factor = 1.65
        status = 'Morning Peak Rush'
        note = 'Historical Morning Office Rush: Expect significant corridor slowdowns (approx +65% travel time).'
    elif (17.5 <= hour <= 21.5):
        factor = 1.85
        status = 'Evening Peak Gridlock'
        note = 'Historical Evening Peak Rush: Heavy commuter density on arterial ring roads (approx +85% travel time).'
    elif (11.5 < hour < 17.5):
        factor = 1.30
        status = 'Moderate City Flow'
        note = 'Historical Mid-day Traffic: Normal urban signal delays.'
    else:
        factor = 1.05
        status = 'Free Flow / Night'
        note = 'Late Night / Early Morning: Low congestion, but increased risk of speeding and signal violations.'

    est_duration = round(base_duration_min * factor, 1)
    delay_min = round(est_duration - base_duration_min, 1)
    return est_duration, status, delay_min, note
