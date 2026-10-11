"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Module: api/weather.py

Live Weather API Integration:
- Primary: OpenWeatherMap (if WEATHER_API_KEY provided in .env)
- Secondary/Free Public Provider: Open-Meteo Weather API (no API key required, live real-time Bengaluru telemetry)
- Graceful Fallback: Historical seasonal Bengaluru weather profile if offline
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', '').strip()
BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

def get_live_weather(lat=BENGALURU_LAT, lon=BENGALURU_LON):
    """
    Fetches real-time weather conditions for Bengaluru coordinates.
    Returns structured dictionary with:
    - temperature_c
    - condition (Clear, Rainy, Foggy, Overcast, Windy)
    - description
    - humidity_pct
    - wind_speed_kmh
    - visibility_km
    - rain_mm
    - is_live: True/False
    - provider_note
    """
    # 1. Try OpenWeatherMap if key is configured
    if WEATHER_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                main = data.get('main', {})
                weather_arr = data.get('weather', [{}])[0]
                wind = data.get('wind', {})
                visibility = data.get('visibility', 10000) / 1000.0
                rain_dict = data.get('rain', {})
                rain_mm = rain_dict.get('1h', 0.0)

                raw_cond = weather_arr.get('main', 'Clear')
                norm_cond = normalize_condition(raw_cond, rain_mm)

                return {
                    'temperature_c': round(main.get('temp', 25.0), 1),
                    'condition': norm_cond,
                    'description': weather_arr.get('description', 'Clear sky').capitalize(),
                    'humidity_pct': int(main.get('humidity', 60)),
                    'wind_speed_kmh': round(wind.get('speed', 3.0) * 3.6, 1),
                    'visibility_km': round(visibility, 1),
                    'rain_mm': round(rain_mm, 1),
                    'is_live': True,
                    'provider': 'OpenWeatherMap API',
                    'status': 'LIVE_ONLINE'
                }
        except Exception as e:
            print(f"[WARN] OpenWeatherMap request failed: {e}")

    # 2. Try Open-Meteo Real-time API (Legitimate open-data scientific weather API, no key required)
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m"
        )
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            cur = data.get('current', {})
            temp = cur.get('temperature_2m', 26.0)
            humidity = cur.get('relative_humidity_2m', 65)
            rain_val = cur.get('rain', 0.0)
            wind_speed = cur.get('wind_speed_10m', 10.0)
            wcode = cur.get('weather_code', 0)

            cond, desc = decode_wmo_code(wcode, rain_val)
            vis = 4.0 if cond in ['Rainy', 'Foggy'] else 10.0

            return {
                'temperature_c': round(float(temp), 1),
                'condition': cond,
                'description': desc,
                'humidity_pct': int(humidity),
                'wind_speed_kmh': round(float(wind_speed), 1),
                'visibility_km': vis,
                'rain_mm': round(float(rain_val), 1),
                'is_live': True,
                'provider': 'Open-Meteo Real-Time Weather API',
                'status': 'LIVE_ONLINE'
            }
    except Exception as e:
        print(f"[WARN] Open-Meteo request failed: {e}")

    # 3. Graceful Fallback: Historical Seasonal Bengaluru Profile
    # Clearly labeled as fallback per requirement 20 & 26
    return {
        'temperature_c': 24.5,
        'condition': 'Clear',
        'description': 'Historical Average Weather Profile (Live API Offline)',
        'humidity_pct': 60,
        'wind_speed_kmh': 12.0,
        'visibility_km': 10.0,
        'rain_mm': 0.0,
        'is_live': False,
        'provider': 'Historical Bengaluru Baseline (Offline Fallback)',
        'status': 'FALLBACK_OFFLINE'
    }

def decode_wmo_code(code, rain):
    """Translates WMO weather codes to our standardized risk features."""
    if rain > 0.5 or code in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99]:
        return 'Rainy', 'Light to Heavy Rain showers'
    elif code in [45, 48]:
        return 'Foggy', 'Fog or dense haze'
    elif code in [1, 2, 3]:
        return 'Overcast', 'Partly cloudy to overcast'
    else:
        return 'Clear', 'Clear skies'

def normalize_condition(raw_cond, rain):
    rc = raw_cond.lower()
    if 'rain' in rc or 'drizzle' in rc or 'thunder' in rc or rain > 0.2:
        return 'Rainy'
    elif 'fog' in rc or 'mist' in rc or 'haze' in rc:
        return 'Foggy'
    elif 'cloud' in rc:
        return 'Overcast'
    elif 'wind' in rc:
        return 'Windy'
    return 'Clear'
