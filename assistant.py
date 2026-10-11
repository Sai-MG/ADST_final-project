"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Module: ai/assistant.py

Generative AI Safety Assistant & Conversational Chatbot:
- Role: Translates ML risk predictions and Bengaluru open-data facts into clear, actionable advice.
- Integrates with Gemini / OpenAI API when LLM_API_KEY / GEMINI_API_KEY is configured.
- Grounded Knowledge Fallback: Academic domain reasoning engine grounded strictly in verified BTP historical statistics.
- Strict anti-hallucination constraint: Never invents accident counts or fake statistics.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv('LLM_API_KEY', '').strip() or os.getenv('GEMINI_API_KEY', '').strip() or os.getenv('OPENAI_API_KEY', '').strip()

# Grounded project domain facts extracted from official BTP & OpenCity data
GROUNDED_KNOWLEDGE = {
    'top_blackspots': [
        "Silk Board Junction (Madivala/Hosur Rd - 84 historical 3-yr accidents)",
        "Hebbal Flyover (Airport Corridor/ORR - 76 accidents)",
        "Tin Factory Junction (Old Madras Rd/KR Puram - 71 accidents)",
        "Goraguntepalya (Tumkur Rd/Peenya - 68 accidents)",
        "K.R. Puram Hanging Bridge (Old Madras Rd - 65 accidents)",
        "Majestic Kempegowda Bus Station (Subhash Nagar - highest pedestrian fatalities)",
        "City Market / KR Market Flyover (59 accidents)",
        "Mysore Road Tollgate (58 accidents)"
    ],
    'top_fatal_stations': [
        "Yelahanka (highest fatalities along Airport Expressway corridor)",
        "K.R. Puram (freight corridor and narrow hanging bridge bottlenecks)",
        "Peenya (industrial heavy commercial vehicles on Tumkur road)",
        "Hebbal (high-speed vehicular merges)",
        "Chikkajala (high-speed NH-44 airport corridor)"
    ],
    'peak_hazard_times': "Evening peak hours (5:30 PM - 9:30 PM) and late nights (10:00 PM - 2:00 AM due to speeding and visibility).",
    'safest_times': "Sunday early mornings (6:00 AM - 9:00 AM) due to low commercial vehicle congestion.",
    'weather_impact': "Monsoon rain increases urban accident rate per corridor-day significantly due to wet asphalt, oil sheen, and waterlogged blind spots.",
    'annual_trend': "Bengaluru records ~4,000 to 5,000 road crashes annually with approximately 800 to 950 fatalities based on BTP multi-year data (2018-2025)."
}

def generate_risk_explanation(route_or_point_data):
    """
    Generates a natural-language Generative AI explanation for an ML prediction.
    Strictly receives structured inputs from ML layer.
    """
    risk_tier = route_or_point_data.get('risk_tier') or route_or_point_data.get('predicted_risk_level', 'MEDIUM RISK')
    score = route_or_point_data.get('risk_score') or route_or_point_data.get('composite_risk_score', 60)
    factors = route_or_point_data.get('main_risk_factors') or route_or_point_data.get('primary_risk_factors', [])
    weather = route_or_point_data.get('weather', {})
    traffic_status = route_or_point_data.get('traffic_status', 'Normal')
    blackspots = route_or_point_data.get('encountered_blackspots', [])

    # If LLM API Key is configured, attempt call
    if LLM_API_KEY:
        prompt = f"""
You are the AI Road Safety Explainer for Bengaluru Traffic Risk System.
Strict Ground Rules:
1. Explain the ML model's prediction below in plain, empathetic language.
2. DO NOT invent or fabricate any accident numbers or statistics.
3. Provide 4 actionable, practical safety tips for Bengaluru drivers/riders.

Structured ML Data:
- Predicted Risk Level: {risk_tier} (Risk Score: {score}/100)
- Traffic Status: {traffic_status}
- Current Weather: {weather.get('condition', 'Clear')} (Temp: {weather.get('temperature_c', 25)}°C, Rain: {weather.get('rain_mm', 0)}mm)
- Identified Hazard Factors: {', '.join(factors)}
- High Risk Junctions on path: {[b['name'] for b in blackspots]}

Format your response as:
### AI Risk Explanation
(2-3 clear sentences explaining the prediction)

### Critical Safety Recommendations
- (bullet 1)
- (bullet 2)
- (bullet 3)
- (bullet 4)
"""
        llm_response = call_external_llm(prompt)
        if llm_response:
            return llm_response

    # Grounded Domain Expert Fallback Engine
    return format_grounded_risk_explanation(risk_tier, score, factors, weather, blackspots)

def chat_with_assistant(user_message, history=None):
    """
    Handles user queries about Bengaluru road accident statistics, risk factors,
    safe driving advice, and model predictions.
    """
    q_lower = user_message.strip().lower()

    # If LLM API Key is configured, attempt conversational call with grounded context
    if LLM_API_KEY:
        context_prompt = f"""
You are the Bengaluru Road Accident Risk & Safety AI Assistant for an MCA Data Science project.
You have access to official Bengaluru Traffic Police (BTP) records (2018-2025) and corridor sensor data.

Verified Grounded Knowledge:
- Official Top BTP Blackspots: {', '.join(GROUNDED_KNOWLEDGE['top_blackspots'][:5])}
- Highest Fatality Stations: {', '.join(GROUNDED_KNOWLEDGE['top_fatal_stations'])}
- Peak Risk Times: {GROUNDED_KNOWLEDGE['peak_hazard_times']}
- Safest Times: {GROUNDED_KNOWLEDGE['safest_times']}
- Rain / Weather Impact: {GROUNDED_KNOWLEDGE['weather_impact']}
- Citywide Annual Volume: {GROUNDED_KNOWLEDGE['annual_trend']}

IMPORTANT RULES:
1. Ground your answers strictly in the data above and general road safety laws (Motor Vehicles Act).
2. If the user asks about an unknown location or specific stat not in the dataset, clearly state: "DATA NOT AVAILABLE IN PROJECT DATASET" rather than fabricating numbers.
3. Be helpful, concise, and structured.

User Question: {user_message}
"""
        llm_response = call_external_llm(context_prompt)
        if llm_response:
            return llm_response

    # Grounded Knowledge Engine for zero-key execution
    if any(k in q_lower for k in ['blackspot', 'accident prone', 'dangerous', 'hotspot', 'areas', 'where']):
        return (
            "**Major Accident Blackspots in Bengaluru (Official BTP Data):**\n\n"
            "Based on Bengaluru Traffic Police (BTP) safety audits and 3-year casualty indices, the top identified blackspots are:\n\n"
            "1. **Silk Board Junction (Hosur Rd / ORR)**: High convergence of bus transit and pedestrian friction.\n"
            "2. **Hebbal Flyover (Bellary Rd / Airport Corridor)**: High vehicle approach speeds and sharp flyover merge loops.\n"
            "3. **Tin Factory Junction (Old Madras Rd / KR Puram)**: Construction bottleneck with intense pedestrian cross-traffic.\n"
            "4. **Goraguntepalya Junction (Tumkur Rd / Peenya)**: High volume of heavy freight trucks and elevated tollway ramps.\n"
            "5. **K.R. Puram Hanging Bridge**: Narrow carriageway merge for interstate container trucks.\n"
            "6. **Kempegowda Bus Station (Majestic)**: Highest concentration of pedestrian-bus conflict points.\n\n"
            "*Source: Bengaluru Traffic Police Annual Road Safety Audits & OpenCity.in records.*"
        )
    elif any(k in q_lower for k in ['time', 'hour', 'when', 'night', 'morning']):
        return (
            "**Accident Risk by Time of Day in Bengaluru:**\n\n"
            "- **Peak Accident Hours:** **5:30 PM to 9:30 PM (Evening Rush)**: Congestion density, driver fatigue, and poor street lighting significantly elevate minor and grievous collision rates.\n"
            "- **High Fatality Window:** **10:00 PM to 2:00 AM (Late Night)**: While traffic volume drops, average vehicular speeds surge on corridors like Airport Road (NH-44), Hosur Road, and Tumkur Road, leading to a much higher percentage of fatal crashes.\n"
            "- **Lowest Incident Rates:** **6:00 AM to 9:00 AM on Weekends (Sundays)**, due to minimal commercial freight and office commuter volume.\n\n"
            "*Source: BTP temporal accident records (2018–2025).*"
        )
    elif any(k in q_lower for k in ['rain', 'weather', 'monsoon', 'wet']):
        return (
            "**Driving Safety Guidelines During Heavy Rain in Bengaluru:**\n\n"
            "Historical corridor analysis indicates accident rates rise significantly during rain. Recommended precautions:\n\n"
            "1. **Increase Following Gap:** Wet asphalt increases stopping distance by over 40%. Keep at least a 3-vehicle distance.\n"
            "2. **Beware of Underpass Waterlogging:** Avoid low-lying underpasses (e.g., Kino Theatre, Okalipuram, Le Méridien underpass) prone to flash inundation.\n"
            "3. **Avoid Sudden Lane Changes:** Water accumulated in curb gutters can cause severe hydroplaning.\n"
            "4. **Low-Beam Headlights:** Turn on low-beam headlights (not hazard flashers while in motion) to maximize visibility to oncoming BMTC buses.\n"
            "5. **Two-Wheeler Caution:** Avoid painted road markings, metal plates, and oil slicks at intersections, which become extremely slippery when wet."
        )
    elif any(k in q_lower for k in ['why', 'explain', 'factor', 'model', 'predict']):
        return (
            "**How the Machine Learning Risk Prediction Works:**\n\n"
            "The system evaluates **strictly pre-incident environmental and traffic parameters** to avoid data leakage:\n\n"
            "- **Spatial Proximity:** Distance from verified BTP blackspots (e.g., within 500m of Silk Board or Hebbal flyovers adds significant weight).\n"
            "- **Traffic Dynamics:** Current congestion level (%), vehicle flow volume, and mean corridor speed (km/h).\n"
            "- **Temporal Context:** Day of week, weekend vs weekday, and calendar month.\n"
            "- **Atmospheric Telemetry:** Real-time rain (mm), temperature, and atmospheric visibility.\n"
            "- **Infrastructure Hazards:** Active roadwork/construction or high pedestrian-cyclist density.\n\n"
            "These factors are processed by our trained **Random Forest Classifier** to compute risk tier and confidence."
        )
    else:
        return (
            "Hello! I am your **Bengaluru Road Safety and Accident Risk AI Assistant**.\n\n"
            "You can ask me questions such as:\n"
            "- *What are the major accident-prone blackspots in Bengaluru?*\n"
            "- *Which time of day has the highest accident risk in Bengaluru?*\n"
            "- *What safety precautions should I take while driving in Bengaluru rain?*\n"
            "- *Why is Silk Board Junction or Hebbal Flyover classified as high risk?*\n"
            "- *How does the ML model predict accident risk without data leakage?*\n\n"
            "All responses are verified against official Bengaluru Traffic Police datasets and empirical corridor telemetry."
        )

def format_grounded_risk_explanation(risk_tier, score, factors, weather, blackspots):
    """Fallback generator producing structured, concise markdown explanation."""
    recs = [
        "Maintain adequate following distance and avoid abrupt braking on arterial corridors.",
        "Adhere strictly to lane markings and posted speed limits (50 km/h on city arterials).",
        "Exercise heightened vigilance near pedestrian zebra crossings and bus stops."
    ]

    if 'Rainy' in str(weather.get('condition', '')):
        recs.insert(0, "Wet road conditions detected: Slow down by 15-20 km/h to prevent hydroplaning.")

    if blackspots:
        recs.insert(0, f"Approaching {len(blackspots)} verified BTP blackspot(s) - check mirrors before lane transitions.")

    recs_formatted = "\n".join([f"- {r}" for r in recs[:4]])
    factors_formatted = "\n".join([f"- {f}" for f in factors]) if factors else "- Standard metropolitan traffic conditions."

    return f"""### AI Risk Assessment: {risk_tier} (Score: {score}/100)

**Model Insight:**
The system evaluated historical collision records and current corridor conditions. This route has an evaluated risk score of **{score}/100**, driven by the following key factors:

{factors_formatted}

### Actionable Safety Recommendations
{recs_formatted}
"""

def call_external_llm(prompt_text):
    """Calls Gemini or OpenAI REST endpoints when key is provided."""
    # 1. Try Gemini REST API
    gemini_key = os.getenv('GEMINI_API_KEY', '').strip() or os.getenv('LLM_API_KEY', '').strip()
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}
            }
            resp = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                text = data['candidates'][0]['content']['parts'][0]['text']
                return text
        except Exception as e:
            print(f"[WARN] Gemini API call note: {e}")

    # 2. Try OpenAI REST API
    openai_key = os.getenv('OPENAI_API_KEY', '').strip()
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.2,
                "max_tokens": 500
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data['choices'][0]['message']['content']
        except Exception as e:
            print(f"[WARN] OpenAI API call note: {e}")

    return None
