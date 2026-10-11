# Bengaluru Road Accident Risk Prediction and Safety Recommendation System Using Machine Learning and Generative AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.9-orange.svg)](https://scikit-learn.org/)
[![Leaflet](https://img.shields.io/badge/Maps-Leaflet.js%20%2F%20OSRM-green.svg)](https://leafletjs.com/)
[![License](https://img.shields.io/badge/License-Academic%20Open%20Source-lightgrey.svg)]()

> **Degree / Program:** Master of Computer Applications (MCA)  
> **Specialization:** Generative AI & Data Science  
> **Geographic Focus:** Bengaluru Metropolitan Area (BBMP Jurisdiction), Karnataka, India

---

## 1. Project Overview & Objectives

Road traffic collisions represent one of the leading causes of preventable fatalities in metropolitan Bengaluru. This system is a **production-style, end-to-end academic Data Science and Generative AI application** that:

1. **Analyzes Official Historical Accident Data**: Ingests multi-year Bengaluru Traffic Police (BTP) station casualty statistics (2018–2025), KSRSAC GIS police station coordinates, and 8,936 real arterial corridor observations.
2. **Predicts Accident Risk Without Data Leakage**: Evaluates pre-incident variables (temporal factors, weather, corridor geometry, congestion levels, blackspot proximity) to classify accident risk tiers (**Low**, **Medium**, **High**).
3. **Integrates Live Telemetry APIs**: Fetches real-time Bengaluru weather (via Open-Meteo & OpenWeatherMap) and computes driving route waypoints and distances (via OSRM).
4. **Delivers Generative AI Safety Explanations**: Uses Generative AI (Gemini / OpenAI with an empirical domain-expert fallback) to translate raw ML scores into empathetic, plain-language driver safety advice.
5. **Interactive Mapping & Voice Interface**: Features a dynamic Leaflet map of Bengaluru blackspots and hands-free Web Speech API (Speech-to-Text & Text-to-Speech).

---

## 2. Real-World Datasets Used (No Fabricated Data)

The system is built strictly on **legitimate, public open datasets**. Complete provenance is documented in [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md).

| Dataset Name | Source / Publisher | Time Period | Size | Description |
|---|---|---|---|---|
| **BTP Station-wise Crashes** | Bengaluru Traffic Police / OpenCity.in | 2018–2025 | Multi-year CSVs | Official fatal, non-fatal, killed, and injured counts across all police station limits |
| **Bengaluru Crashes Trend** | Bengaluru Traffic Police | 2007–2025 | 19 years | Citywide historical annual accident totals |
| **BTP Traffic Stations KML** | KSRSAC / Karnataka State GIS | 2025 | 43 Stations | Official WGS84 GPS coordinates (Lat, Lon) for all BTP traffic stations |
| **Bengaluru Corridor Traffic** | Open Data Science Research | 2022–2023 | 8,936 rows | Daily arterial corridor telemetry: volume, speed, weather, congestion, and incident counts |
| **BTP Blackspots Reference** | BTP Road Safety Audits & MoRTH | 3-Year Audits | 30 Hotspots | Verified high-casualty blackspots (Silk Board, Hebbal, Tin Factory, Goraguntepalya, etc.) |

---

## 3. Machine Learning Architecture & Zero Data Leakage

### A. Prevention of Data Leakage
To maintain strict academic integrity, variables that are only known **AFTER** a collision occurs (e.g., number of vehicles damaged, injuries, fatalities, crash severity) are **strictly excluded** from model training inputs.

* **Predictor Features ($X$)**:
  * `traffic_volume`, `average_speed`, `congestion_level`, `travel_time_index`
  * `road_capacity_utilization`, `roadwork_present`, `pedestrian_cyclist_count`
  * `distance_to_nearest_blackspot_km`, `latitude`, `longitude`
  * `month`, `day_of_week`, `is_weekend`
  * `weather_condition`, `area_name`
* **Target Variable ($y$)**: `risk_level` (0: Low Risk [0 incidents], 1: Medium Risk [1–2 incidents], 2: High Risk [$\ge 3$ incidents]).

### B. Benchmarked Classifiers Comparison (Test Set: 1,788 Samples)
| Model Algorithm | Accuracy | Weighted Precision | Macro F1-Score | Weighted F1-Score | ROC-AUC | Decision |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 41.05% | 48.81% | 0.3964 | 0.3320 | 0.6569 | Baseline |
| **Decision Tree** | 42.39% | 46.97% | 0.4341 | 0.4033 | 0.6238 | Prone to variance |
| **Random Forest** | **46.87%** | **52.44%** | **0.4823** | **0.4598** | **0.6664** | **Selected Optimal** |
| **Gradient Boosting** | 54.59% | 52.09% | 0.4583 | 0.5073 | 0.6678 | Biased to majority class |

> **Selection Rationale:** Random Forest was selected as the optimal model because it achieved the **highest balanced Macro F1-Score (0.4823)** across all three risk categories, successfully identifying rare high-risk hazards without being biased toward the majority class.

---

## 4. Project Directory Structure

```text
bengaluru-accident-risk/
│
├── app.py                      # Main Flask application and REST APIs
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive documentation
├── run_project.bat             # One-click startup script for Windows
├── .env.example                # Configuration template
├── .gitignore                  # Git exclusion rules
│
├── data/
│   ├── raw/                    # Raw official downloads (BTP CSVs, KML)
│   ├── processed/              # Cleaned dataset & preprocessing report
│   ├── download_data.py        # Automated dataset downloader
│   ├── preprocess.py           # Data cleaning & feature engineering pipeline
│   └── DATA_SOURCES.md         # Full provenance & metadata documentation
│
├── ml/
│   ├── train_model.py          # ML benchmarking & model training script
│   ├── evaluate_model.py       # Evaluation metrics & scenario testing
│   └── prediction.py           # Inference engine for points & routes
│
├── models/
│   ├── accident_risk_model.pkl # Trained Random Forest pipeline artifact
│   ├── preprocessor.pkl        # Scaler & OneHotEncoder pipeline
│   └── model_metadata.json     # Performance metrics & feature rankings
│
├── database/
│   ├── db.py                   # SQLite manager (Auth, Predictions, Chats)
│   └── accident_risk.db        # SQLite database file
│
├── api/
│   ├── weather.py              # OpenWeather & Open-Meteo live API integration
│   ├── traffic.py              # OSRM routing, Nominatim, & TomTom API
│   └── maps.py                 # Leaflet GeoJSON providers & route risk evaluator
│
├── ai/
│   └── assistant.py            # Generative AI explainer & safety chatbot
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Shared layout & navigation
│   ├── index.html              # Home landing page
│   ├── dashboard.html          # Interactive KPI & Chart.js dashboard
│   ├── route_prediction.html   # Route evaluator & Leaflet path map
│   ├── chatbot.html            # AI chat interface with voice controls
│   ├── login.html              # User authentication
│   ├── register.html           # User registration
│   ├── forgot_password.html    # Password recovery request
│   └── reset_password.html     # OTP reset form
│
├── static/
│   ├── css/style.css           # Custom Data Science styling
│   └── js/
│       ├── main.js             # Chart.js analytics engine
│       ├── map.js              # Leaflet mapping engine
│       └── voice.js            # Web Speech API (STT & TTS)
│
├── notebooks/
│   └── eda_analysis.py         # Exploratory Data Analysis script
│
├── outputs/figures/            # Generated figures (charts & confusion matrix)
│
└── tests/                      # Automated test suite (20 tests)
    ├── test_data.py            # Dataset integrity tests
    ├── test_model.py           # ML inference tests
    └── test_routes.py          # Integration & API failure tests
```

---

## 5. Quick Start (Windows One-Click)

The easiest way to run the project on Windows:

1. Double-click **`run_project.bat`** in the project root folder.
2. The batch script will automatically:
   - Create a Python virtual environment (`venv`) if needed.
   - Activate the environment.
   - Install all required dependencies from `requirements.txt`.
   - Verify datasets and train models automatically if missing.
   - Launch the Flask server at `http://127.0.0.1:5000`.

---

## 6. Manual Step-by-Step Installation

### Step 1: Clone or Navigate to Directory
```bash
cd "C:\Users\Saimuthu Genesh\.gemini\antigravity\scratch\bengaluru-accident-risk"
```

### Step 2: Create & Activate Virtual Environment
**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Datasets & Run Preprocessing
```bash
python data/download_data.py
python data/preprocess.py
```
*Output: `data/processed/bengaluru_accidents_clean.csv` (8,936 rows)*

### Step 5: Run Exploratory Data Analysis (EDA)
```bash
python notebooks/eda_analysis.py
```
*Generates 7 charts saved into `outputs/figures/`.*

### Step 6: Train Machine Learning Models
```bash
python ml/train_model.py
```
*Benchmarks Logistic Regression, Decision Tree, Random Forest, Gradient Boosting; saves `models/accident_risk_model.pkl`.*

### Step 7: Evaluate Models & Run Scenarios
```bash
python ml/evaluate_model.py
```

### Step 8: Run Automated Test Suite
```bash
python -m unittest discover tests
```
*Runs all 20 data, model, and route tests.*

### Step 9: Launch Web Application
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 7. Demo Academic Credentials

For evaluation and demonstration purposes, a test account is pre-registered in the database:

* **Email:** `student@rvce.edu.in`
* **Password:** `Student@2026`

*You can also create a new account via the **Sign Up** page.*

---

## 8. API Keys & Zero-Key Fallback Design

The application is engineered to work **immediately without any paid API keys**:

| Service | Config Variable in `.env` | Default Zero-Key Behavior |
|---|---|---|
| **Live Weather** | `WEATHER_API_KEY` | Uses **Open-Meteo Real-time API** (free, open, no key required). If offline, seamlessly uses Bengaluru historical seasonal profile. |
| **Route Geometry** | `TRAFFIC_API_KEY` | Uses **OSRM (Open Source Routing Machine)** for driving geometry and waypoints; evaluates congestion via empirical historical peak/off-peak models. |
| **Live Traffic** | `TRAFFIC_API_KEY` | Supports TomTom Traffic API. If omitted, clearly displays: *“Live traffic congestion data unavailable (Using historical corridor model).”* |
| **Generative AI** | `GEMINI_API_KEY` or `LLM_API_KEY` | Integrates Google Gemini 1.5 Flash. If key is omitted, uses the built-in **Academic Grounded Knowledge Engine** grounded in official BTP facts. |

To add your own API keys, simply copy `.env.example` to `.env` and fill in the values:
```bash
cp .env.example .env
```

---

## 9. Troubleshooting & FAQ

### Q1: `ModuleNotFoundError` when running scripts
**Solution:** Ensure your virtual environment is activated (`venv\Scripts\activate`) before running python scripts.

### Q2: Port 5000 is already in use
**Solution:** Change `PORT=5001` in your `.env` file or run:
```bash
python -c "from app import app; app.run(port=5001)"
```

### Q3: Why does route prediction work without an external Google Maps API key?
**Solution:** The system uses OSRM (Open Source Routing Machine) which calculates authentic turn-by-turn driving coordinates using OpenStreetMap network data for Bengaluru.

### Q4: Voice Speech-to-Text does not respond
**Solution:** Web Speech API requires browser microphone permissions. Click "Allow" when the browser requests microphone access. Text input is always available as a direct alternative.
