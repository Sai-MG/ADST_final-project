"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Script: data/preprocess.py

End-to-end data cleaning, feature engineering, and preprocessing pipeline.
Outputs:
- data/processed/bengaluru_accidents_clean.csv
- data/processed/btp_station_risk_summary.csv
- data/processed/preprocessing_report.json
"""

import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# High-precision coordinates for Bengaluru corridors in the dataset
CORRIDOR_COORDINATES = {
    ('Electronic City', 'Hosur Road'): (12.8452, 77.6602),
    ('Electronic City', 'Silk Board Junction'): (12.9174, 77.6238),
    ('Hebbal', 'Ballari Road'): (13.0189, 77.5882),
    ('Hebbal', 'Hebbal Flyover'): (13.0358, 77.5970),
    ('Indiranagar', '100 Feet Road'): (12.9719, 77.6412),
    ('Indiranagar', 'CMH Road'): (12.9792, 77.6401),
    ('Jayanagar', 'Jayanagar 4th Block'): (12.9298, 77.5826),
    ('Jayanagar', 'South End Circle'): (12.9382, 77.5801),
    ('Koramangala', 'Sarjapur Road'): (12.9238, 77.6432),
    ('Koramangala', 'Sony World Junction'): (12.9345, 77.6266),
    ('M.G. Road', 'Anil Kumble Circle'): (12.9756, 77.6067),
    ('M.G. Road', 'Trinity Circle'): (12.9734, 77.6205),
    ('Whitefield', 'ITPL Main Road'): (12.9856, 77.7315),
    ('Whitefield', 'Marathahalli Bridge'): (12.9562, 77.7011),
    ('Yeshwanthpur', 'Tumkur Road'): (13.0285, 77.5408),
    ('Yeshwanthpur', 'Yeshwanthpur Circle'): (13.0213, 77.5501),
}

def parse_btp_kml():
    """Parses official BTP Traffic Police Stations KML file for GIS coordinates."""
    kml_path = os.path.join(RAW_DIR, 'bengaluru_traffic_police_stations.kml')
    stations = []
    if not os.path.exists(kml_path):
        return pd.DataFrame()

    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        for p in root.iter('{http://www.opengis.net/kml/2.2}Placemark'):
            name = None
            code = None
            for sd in p.iter('{http://www.opengis.net/kml/2.2}SimpleData'):
                if sd.get('name') == 'TRF_POL_STAName':
                    name = sd.text.strip() if sd.text else ''
                elif sd.get('name') == 'KGISTPSCode':
                    code = sd.text.strip() if sd.text else ''
            coord_elem = p.find('.//{http://www.opengis.net/kml/2.2}coordinates')
            if coord_elem is not None and coord_elem.text:
                parts = coord_elem.text.strip().split(',')
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    stations.append({
                        'station_name': name.replace('Traffic Police Station, ', '').strip() if name else 'Unknown',
                        'station_code': code or '',
                        'latitude': lat,
                        'longitude': lon
                    })
    except Exception as e:
        print(f"[WARN] Error reading KML: {e}")

    return pd.DataFrame(stations)

def calculate_haversine(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two GPS points in kilometers."""
    R = 6371.0 # Earth's radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def preprocess_pipeline():
    print("=" * 70)
    print("STARTING BENGALURU ACCIDENT DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    report = {
        'pipeline_status': 'IN_PROGRESS',
        'raw_datasets': [],
        'transformations': []
    }

    # 1. Load Corridor Dataset
    raw_csv = os.path.join(RAW_DIR, 'Banglore_traffic_Dataset.csv')
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Missing raw dataset: {raw_csv}. Run download_data.py first.")

    df_raw = pd.read_csv(raw_csv)
    original_row_count = len(df_raw)
    report['original_row_count'] = original_row_count
    print(f"[STEP 1] Loaded raw corridor dataset: {original_row_count:,} rows, {df_raw.shape[1]} columns")

    # 2. Check & Handle Duplicates
    duplicates_count = int(df_raw.duplicated().sum())
    report['removed_duplicates'] = duplicates_count
    df = df_raw.drop_duplicates().copy()
    if duplicates_count > 0:
        print(f"[STEP 2] Removed {duplicates_count} duplicate rows")
        report['transformations'].append(f"Dropped {duplicates_count} duplicates")
    else:
        print(f"[STEP 2] Zero duplicates found")

    # 3. Check Missing Values
    null_counts = df.isnull().sum().to_dict()
    report['missing_values_detected'] = {k: int(v) for k, v in null_counts.items() if v > 0}
    print(f"[STEP 3] Missing values checked: {sum(null_counts.values())} missing values")

    # Impute or fill if any missing values existed
    if df['Traffic Volume'].isnull().any():
        df['Traffic Volume'] = df['Traffic Volume'].fillna(df['Traffic Volume'].median())
    if df['Average Speed'].isnull().any():
        df['Average Speed'] = df['Average Speed'].fillna(df['Average Speed'].median())
    if df['Weather Conditions'].isnull().any():
        df['Weather Conditions'] = df['Weather Conditions'].fillna('Clear')

    # 4. Standardize & Parse Temporal Attributes
    # Dataset uses DD-MM-YYYY format
    df['parsed_date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    # If any failed, try standard parser
    if df['parsed_date'].isnull().any():
        df['parsed_date'] = df['parsed_date'].fillna(pd.to_datetime(df['Date'], errors='coerce'))

    df['year'] = df['parsed_date'].dt.year
    df['month'] = df['parsed_date'].dt.month
    df['day'] = df['parsed_date'].dt.day
    df['day_of_week'] = df['parsed_date'].dt.dayofweek # 0=Monday, 6=Sunday
    df['day_name'] = df['parsed_date'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'

    df['season'] = df['month'].apply(get_season)
    report['transformations'].append("Engineered temporal features: year, month, day, day_of_week, day_name, is_weekend, season")
    print(f"[STEP 4] Parsed dates and engineered temporal features")

    # 5. Geocode Corridors with Accurate WGS84 Coordinates
    def assign_coords(row):
        key = (str(row['Area Name']).strip(), str(row['Road/Intersection Name']).strip())
        if key in CORRIDOR_COORDINATES:
            return CORRIDOR_COORDINATES[key]
        return (12.9716, 77.5946) # Fallback: Bengaluru Central (Vidhana Soudha)

    coords = df.apply(assign_coords, axis=1)
    df['latitude'] = [c[0] for c in coords]
    df['longitude'] = [c[1] for c in coords]
    report['transformations'].append("Mapped corridors to exact geodetic coordinates (latitude, longitude)")
    print(f"[STEP 5] Geocoded {len(df)} corridor records with WGS84 coordinates")

    # 6. Normalize Categories
    # Standardize Weather Conditions
    weather_map = {
        'Clear': 'Clear',
        'Rain': 'Rainy',
        'Fog': 'Foggy',
        'Overcast': 'Overcast',
        'Windy': 'Windy'
    }
    df['weather_condition'] = df['Weather Conditions'].map(weather_map).fillna('Clear')

    # Road Condition Feature Synthesis (Pre-accident indicator)
    def determine_road_condition(row):
        is_construction = str(row['Roadwork and Construction Activity']).strip().lower() == 'yes'
        weather = str(row['weather_condition']).lower()
        capacity_util = row['Road Capacity Utilization']
        
        if is_construction:
            return 'Under Construction'
        elif 'rain' in weather:
            return 'Wet/Slippery'
        elif capacity_util > 90:
            return 'Heavy Bottleneck'
        else:
            return 'Normal Dry'

    df['road_condition'] = df.apply(determine_road_condition, axis=1)
    df['roadwork_present'] = df['Roadwork and Construction Activity'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    report['transformations'].append("Standardized weather condition and synthesized pre-incident road_condition indicator")
    print(f"[STEP 6] Standardized categorical indicators: weather and road conditions")

    # 7. Spatial Blackspot Proximity Feature
    blackspots_path = os.path.join(RAW_DIR, 'bengaluru_blackspots.csv')
    if os.path.exists(blackspots_path):
        df_blackspots = pd.read_csv(blackspots_path)
        def dist_to_nearest_blackspot(lat, lon):
            distances = [calculate_haversine(lat, lon, blat, blon) for blat, blon in zip(df_blackspots['latitude'], df_blackspots['longitude'])]
            return min(distances) if distances else 0.0

        df['distance_to_nearest_blackspot_km'] = [
            dist_to_nearest_blackspot(lat, lon) for lat, lon in zip(df['latitude'], df['longitude'])
        ]
        report['transformations'].append("Calculated spatial distance to nearest BTP official accident blackspot (km)")
        print(f"[STEP 7] Computed spatial blackspot proximity distances")

    # 8. Define Risk Target Variables (Strictly separated from pre-incident inputs)
    # Target 1: Incident Count (raw integer: 0 to 10)
    # Target 2: Multi-class Risk Level
    #   - 0: Low Risk (0 incidents)
    #   - 1: Medium Risk (1 to 2 incidents)
    #   - 2: High Risk (>= 3 incidents)
    # Target 3: Binary High Risk Indicator (0: Low/Medium, 1: High)
    # Target 4: Normalized Continuous Risk Score (0 to 100)
    
    def classify_risk_tier(incidents):
        if incidents == 0:
            return 'Low'
        elif incidents <= 2:
            return 'Medium'
        else:
            return 'High'

    def classify_risk_level_id(incidents):
        if incidents == 0:
            return 0
        elif incidents <= 2:
            return 1
        else:
            return 2

    df['risk_tier'] = df['Incident Reports'].apply(classify_risk_tier)
    df['risk_level'] = df['Incident Reports'].apply(classify_risk_level_id)
    df['is_high_risk'] = df['risk_level'].apply(lambda x: 1 if x == 2 else 0)

    # Risk Score Calculation (0-100 composite index)
    # Formula combines historical incident density, congestion index, and capacity saturation
    incident_norm = np.clip(df['Incident Reports'] / 6.0, 0, 1) * 50
    congestion_norm = (df['Congestion Level'] / 100.0) * 30
    capacity_norm = (df['Road Capacity Utilization'] / 100.0) * 20
    df['risk_score'] = np.round(incident_norm + congestion_norm + capacity_norm, 1)

    # Rename & clean column names to snake_case
    rename_dict = {
        'Area Name': 'area_name',
        'Road/Intersection Name': 'intersection_name',
        'Traffic Volume': 'traffic_volume',
        'Average Speed': 'average_speed',
        'Travel Time Index': 'travel_time_index',
        'Congestion Level': 'congestion_level',
        'Road Capacity Utilization': 'road_capacity_utilization',
        'Incident Reports': 'incident_reports',
        'Environmental Impact': 'environmental_impact',
        'Public Transport Usage': 'public_transport_usage',
        'Traffic Signal Compliance': 'traffic_signal_compliance',
        'Parking Usage': 'parking_usage',
        'Pedestrian and Cyclist Count': 'pedestrian_cyclist_count'
    }
    df = df.rename(columns=rename_dict)

    # 9. Clean Police Station Jurisdiction Summary
    stations_kml_df = parse_btp_kml()
    summary_path = os.path.join(PROCESSED_DIR, 'btp_station_risk_summary.csv')
    
    # Process BTP 2024 & 2023 station crashes
    btp_2024_path = os.path.join(RAW_DIR, 'btp_2024_station_wise.csv')
    if os.path.exists(btp_2024_path):
        df_btp24 = pd.read_csv(btp_2024_path)
        # Clean columns
        df_btp24.columns = [c.strip() for c in df_btp24.columns]
        # Match with KML stations if available
        if not stations_kml_df.empty:
            merged_stations = pd.merge(
                df_btp24, stations_kml_df,
                left_on='Station', right_on='station_name',
                how='left'
            )
            merged_stations['latitude'] = merged_stations['latitude'].fillna(12.9716)
            merged_stations['longitude'] = merged_stations['longitude'].fillna(77.5946)
            merged_stations.to_csv(summary_path, index=False)
            print(f"[STEP 8] Created BTP Police Station Risk Summary ({len(merged_stations)} stations) -> {summary_path}")
        else:
            df_btp24.to_csv(summary_path, index=False)

    # 10. Save Final Cleaned Dataset
    clean_csv_path = os.path.join(PROCESSED_DIR, 'bengaluru_accidents_clean.csv')
    df.to_csv(clean_csv_path, index=False)
    final_row_count = len(df)
    report['final_row_count'] = final_row_count
    report['final_columns'] = df.columns.tolist()
    report['risk_tier_distribution'] = df['risk_tier'].value_counts().to_dict()
    report['weather_distribution'] = df['weather_condition'].value_counts().to_dict()
    report['corridor_count'] = int(df['intersection_name'].nunique())
    report['clean_csv_path'] = clean_csv_path
    report['pipeline_status'] = 'COMPLETED_SUCCESS'

    # Save preprocessing report
    report_json_path = os.path.join(PROCESSED_DIR, 'preprocessing_report.json')
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)

    print("=" * 70)
    print(f"PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"Original Rows: {original_row_count:,} | Final Cleaned Rows: {final_row_count:,}")
    print(f"Duplicates Removed: {duplicates_count} | Columns: {len(df.columns)}")
    print(f"Risk Tier Breakdown:")
    for tier, cnt in report['risk_tier_distribution'].items():
        print(f"   - {tier:8s}: {cnt:,} ({cnt/final_row_count*100:.1f}%)")
    print(f"Cleaned dataset saved to: {clean_csv_path}")
    print(f"Preprocessing report: {report_json_path}")
    print("=" * 70)

    return df, report

if __name__ == '__main__':
    preprocess_pipeline()
