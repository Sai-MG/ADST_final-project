"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Main Flask Application (app.py)

Academic Data Science & Generative AI Web Application.
"""

import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from dotenv import load_dotenv

from database.db import (
    init_db, register_user, authenticate_user,
    generate_password_reset_otp, reset_password_with_otp,
    save_route_prediction, save_chat_log, get_recent_predictions
)
from api.weather import get_live_weather
from api.traffic import geocode_location, get_route, BENGALURU_LANDMARKS
from api.maps import get_blackspots_geojson, get_btp_stations_geojson, evaluate_route_risk
from ml.prediction import get_predictor
from ai.assistant import generate_risk_explanation, chat_with_assistant

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'bengaluru-accident-risk-secret-key-2026')

# Initialize DB on startup
with app.app_context():
    init_db()

# --- Helper Context Processor ---
@app.context_processor
def inject_global_data():
    weather = get_live_weather()
    return {
        'current_user': session.get('user'),
        'current_weather': weather,
        'app_year': datetime.now().year
    }

# ==========================================
# PAGE ROUTES
# ==========================================

@app.route('/')
def home():
    predictor = get_predictor()
    meta = predictor.metadata or {}
    blackspots = predictor.blackspots[:6]
    return render_template('index.html', metadata=meta, sample_blackspots=blackspots)

@app.route('/dashboard')
def dashboard():
    predictor = get_predictor()
    meta = predictor.metadata or {}
    recent_searches = get_recent_predictions(session.get('user', {}).get('id'), limit=5)
    return render_template('dashboard.html', metadata=meta, recent_searches=recent_searches)

@app.route('/predict-route', methods=['GET'])
def route_prediction_page():
    origin = request.args.get('origin', 'Majestic')
    destination = request.args.get('destination', 'Whitefield')
    landmarks = sorted([v[2] for v in BENGALURU_LANDMARKS.values()])
    return render_template('route_prediction.html', default_origin=origin, default_destination=destination, landmarks=landmarks)

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

# ==========================================
# AUTHENTICATION ROUTES
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        success, user = authenticate_user(email, password)
        if success:
            session['user'] = {
                'id': user['id'],
                'email': user['email'],
                'full_name': user['full_name']
            }
            flash(f"Welcome back, {user['full_name']}!", 'success')
            next_url = request.args.get('next') or url_for('dashboard')
            return redirect(next_url)
        else:
            flash("Invalid email address or password.", 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_pwd = request.form.get('confirm_password', '').strip()

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", 'warning')
            return render_template('register.html')
        if password != confirm_pwd:
            flash("Passwords do not match.", 'warning')
            return render_template('register.html')

        success, res = register_user(email, password, full_name)
        if success:
            flash("Account successfully created! You may now log in.", 'success')
            return redirect(url_for('login'))
        else:
            flash(res, 'danger')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been securely logged out.", 'info')
    return redirect(url_for('home'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    dev_otp_notice = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        success, otp_or_msg = generate_password_reset_otp(email)
        if success:
            session['reset_email'] = email
            dev_otp_notice = f"DEVELOPMENT MODE NOTICE: No external SMS/SMTP gateway connected. Your simulated OTP is: {otp_or_msg}"
            flash("Verification OTP generated. Please enter your code to reset password.", 'info')
            return render_template('reset_password.html', email=email, dev_otp_notice=dev_otp_notice)
        else:
            flash(otp_or_msg, 'danger')
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['POST'])
def reset_password():
    email = request.form.get('email', '').strip() or session.get('reset_email', '')
    otp = request.form.get('otp', '').strip()
    new_password = request.form.get('new_password', '').strip()

    if len(new_password) < 6:
        flash("Password must be at least 6 characters.", 'warning')
        return render_template('reset_password.html', email=email)

    success, msg = reset_password_with_otp(email, otp, new_password)
    if success:
        session.pop('reset_email', None)
        flash(msg, 'success')
        return redirect(url_for('login'))
    else:
        flash(msg, 'danger')
        return render_template('reset_password.html', email=email)

# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route('/api/predict-route', methods=['POST'])
def api_predict_route():
    try:
        data = request.get_json() or {}
        origin = data.get('origin')
        destination = data.get('destination')

        if not origin or not destination or not str(origin).strip() or not str(destination).strip():
            return jsonify({'error': 'Please provide both origin and destination'}), 400

        origin = str(origin).strip()
        destination = str(destination).strip()

        # 1. Run Route Risk Evaluation Engine
        route_eval = evaluate_route_risk(origin, destination)

        # 2. Run Generative AI Explanation
        ai_explanation = generate_risk_explanation({
            'risk_tier': route_eval['overall_risk']['risk_tier'],
            'risk_score': route_eval['overall_risk']['risk_score'],
            'main_risk_factors': route_eval['overall_risk']['main_risk_factors'],
            'traffic_status': route_eval['route']['traffic_status'],
            'weather': route_eval['weather'],
            'encountered_blackspots': route_eval['overall_risk']['encountered_blackspots']
        })

        route_eval['ai_explanation'] = ai_explanation

        # 3. Save to database
        user_id = session.get('user', {}).get('id')
        save_route_prediction(
            user_id=user_id,
            origin=route_eval['origin']['name'],
            destination=route_eval['destination']['name'],
            distance=route_eval['route']['distance_km'],
            risk_tier=route_eval['overall_risk']['risk_tier'],
            risk_score=route_eval['overall_risk']['risk_score'],
            traffic=route_eval['route']['traffic_status'],
            weather=route_eval['weather']['condition']
        )

        return jsonify(route_eval)

    except Exception as e:
        print(f"[API ERROR] Route prediction error: {e}")
        return jsonify({
            'error': f'Route risk evaluation failed: {str(e)}',
            'fallback_message': 'Historical accident statistics are still available on the dashboard.'
        }), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json() or {}
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        bot_reply = chat_with_assistant(user_message)
        
        # Save to database
        user_id = session.get('user', {}).get('id')
        save_chat_log(user_id, user_message, bot_reply)

        return jsonify({
            'reply': bot_reply,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': f'Chatbot service error: {str(e)}'}), 500

@app.route('/api/blackspots', methods=['GET'])
def api_blackspots():
    geojson = get_blackspots_geojson()
    return jsonify(geojson)

@app.route('/api/btp-stations', methods=['GET'])
def api_btp_stations():
    geojson = get_btp_stations_geojson()
    return jsonify(geojson)

@app.route('/api/weather', methods=['GET'])
def api_weather():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat and lon:
        w = get_live_weather(lat, lon)
    else:
        w = get_live_weather()
    return jsonify(w)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Returns aggregated stats for charts on the dashboard."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_csv = os.path.join(base_dir, 'data', 'processed', 'bengaluru_accidents_clean.csv')
    
    if not os.path.exists(clean_csv):
        return jsonify({'error': 'Cleaned dataset not found'}), 404

    import pandas as pd
    df = pd.read_csv(clean_csv)

    # 1. Monthly Trends
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly = df.groupby('month')['incident_reports'].sum().reindex(range(1, 13), fill_value=0).tolist()

    # 2. Day of Week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df.groupby('day_name')['incident_reports'].sum().reindex(day_order, fill_value=0).tolist()

    # 3. Corridor Incidents
    corridors = df.groupby('intersection_name')['incident_reports'].sum().sort_values(ascending=False).head(8)

    # 4. Weather Impact
    weather_impact = df.groupby('weather_condition')['incident_reports'].mean().round(2).to_dict()

    # 5. Risk Tier Breakdown
    risk_tiers = df['risk_tier'].value_counts().to_dict()

    return jsonify({
        'total_records': len(df),
        'total_incidents': int(df['incident_reports'].sum()),
        'monthly_labels': month_names,
        'monthly_data': monthly,
        'daily_labels': day_order,
        'daily_data': daily,
        'corridor_labels': corridors.index.tolist(),
        'corridor_data': corridors.values.tolist(),
        'weather_impact': weather_impact,
        'risk_tiers': risk_tiers
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting Bengaluru Accident Risk Web Application on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
