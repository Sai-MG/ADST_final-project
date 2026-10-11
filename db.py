"""
Bengaluru Road Accident Risk Prediction and Safety Recommendation System
Module: database/db.py

SQLite Database layer for User Authentication, Route Risk History, Searches, and AI Chat logs.
"""

import os
import sqlite3
import random
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'accident_risk.db')

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        reset_otp TEXT,
        otp_expiry TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS route_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        origin TEXT NOT NULL,
        destination TEXT NOT NULL,
        distance_km REAL,
        risk_tier TEXT NOT NULL,
        risk_score INTEGER NOT NULL,
        traffic_status TEXT,
        weather_condition TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS search_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        search_type TEXT,
        query TEXT NOT NULL,
        result_summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS ai_chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    # Create demo academic test user if not present
    cursor.execute("SELECT id FROM users WHERE email = 'student@rvce.edu.in'")
    if not cursor.fetchone():
        demo_pwd_hash = generate_password_hash("Student@2026")
        cursor.execute("""
            INSERT INTO users (email, password_hash, full_name)
            VALUES ('student@rvce.edu.in', ?, 'MCA Scholar')
        """, (demo_pwd_hash,))

    conn.commit()
    conn.close()

def register_user(email, password, full_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email.lower().strip(),))
    if cursor.fetchone():
        conn.close()
        return False, "An account with this email address already exists."

    pwd_hash = generate_password_hash(password)
    cursor.execute("""
        INSERT INTO users (email, password_hash, full_name)
        VALUES (?, ?, ?)
    """, (email.lower().strip(), pwd_hash, full_name.strip()))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return True, user_id

def authenticate_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, password_hash, full_name FROM users WHERE email = ?", (email.lower().strip(),))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        return True, dict(user)
    return False, None

def generate_password_reset_otp(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email.lower().strip(),))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return False, "Email address not registered."

    otp = str(random.randint(100000, 999999))
    expiry = datetime.now() + timedelta(minutes=15)

    cursor.execute("""
        UPDATE users SET reset_otp = ?, otp_expiry = ? WHERE id = ?
    """, (otp, expiry.strftime('%Y-%m-%d %H:%M:%S'), user['id']))
    conn.commit()
    conn.close()
    return True, otp

def reset_password_with_otp(email, otp, new_password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, otp_expiry FROM users WHERE email = ? AND reset_otp = ?
    """, (email.lower().strip(), str(otp).strip()))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return False, "Invalid OTP or email address."

    # Verify expiry
    expiry = datetime.strptime(user['otp_expiry'], '%Y-%m-%d %H:%M:%S')
    if datetime.now() > expiry:
        conn.close()
        return False, "OTP has expired. Please request a new verification code."

    new_hash = generate_password_hash(new_password)
    cursor.execute("""
        UPDATE users SET password_hash = ?, reset_otp = NULL, otp_expiry = NULL WHERE id = ?
    """, (new_hash, user['id']))
    conn.commit()
    conn.close()
    return True, "Password reset successfully. You can now login."

def save_route_prediction(user_id, origin, destination, distance, risk_tier, risk_score, traffic, weather):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO route_predictions (user_id, origin, destination, distance_km, risk_tier, risk_score, traffic_status, weather_condition)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, origin, destination, distance, risk_tier, risk_score, traffic, weather))
    conn.commit()
    conn.close()

def save_chat_log(user_id, question, answer):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ai_chat_logs (user_id, question, answer)
        VALUES (?, ?, ?)
    """, (user_id, question, answer))
    conn.commit()
    conn.close()

def get_recent_predictions(user_id=None, limit=10):
    conn = get_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute("""
            SELECT * FROM route_predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
        """, (user_id, limit))
    else:
        cursor.execute("""
            SELECT * FROM route_predictions ORDER BY created_at DESC LIMIT ?
        """, (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully at:", DB_PATH)
