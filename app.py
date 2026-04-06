from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import joblib
import pandas as pd
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'glucobreath_secret_key'  # For session management
MODEL_FILE = 'models/gluco_model.pkl'
DB_FILE = 'data/users.db'

# Initialize Model
model = None
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)

# Setup Simple User DB (SQLite)
def init_db():
    if not os.path.exists('data'):
        os.makedirs('data')
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS history 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      user_id INTEGER, 
                      acetone REAL, 
                      temp REAL, 
                      humidity REAL, 
                      prediction REAL, 
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# --- Routes ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session['user'] = username
            return redirect(url_for('dashboard'))
        return "Invalid Credentials", 401
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')
        hashed_pw = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists!", 400
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    global model
    if model is None:
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
        else:
            return jsonify({'error': 'Model not trained!'}), 500

    try:
        data = request.get_json()
        acetone = float(data['acetone'])
        temp = float(data['temperature'])
        hum = float(data['humidity'])
        
        input_df = pd.DataFrame([[acetone, temp, hum]], 
                                  columns=['Acetone', 'Temperature', 'Humidity'])
        prediction = model.predict(input_df)[0]
        prediction = round(float(prediction), 2)
        
        # Save to History
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username=?", (session['user'],))
        u_id = cursor.fetchone()[0]
        cursor.execute("INSERT INTO history (user_id, acetone, temp, humidity, prediction) VALUES (?, ?, ?, ?, ?)",
                       (u_id, acetone, temp, hum, prediction))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_history')
def get_history():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username=?", (session['user'],))
    u_id = cursor.fetchone()[0]
    cursor.execute("SELECT acetone, temp, humidity, prediction, timestamp FROM history WHERE user_id=? ORDER BY id DESC LIMIT 10", (u_id,))
    rows = cursor.fetchall()
    conn.close()
    
    history_list = []
    for r in rows:
        history_list.append({'acetone': r[0], 'temp': r[1], 'humidity': r[2], 'prediction': r[3], 'time': r[4]})
    
    return jsonify({'history': history_list})

import requests

@app.route('/fetch_sensor_data')
def fetch_sensor_data():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    sensor_ip = request.args.get('ip')
    if not sensor_ip:
        return jsonify({'error': 'No IP provided'}), 400
    
    # Ensure it starts with http
    if not sensor_ip.startswith('http'):
        sensor_ip = 'http://' + sensor_ip

    try:
        # We expect the ESP32 to return JSON: {"acetone": 12.5, "temp": 31, "hum": 55}
        # Timeout set to 2s to prevent server hang
        response = requests.get(sensor_ip, timeout=5)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': f"Failed to connect to {sensor_ip}: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Starting GlucoBreath Enterprise Web App...")
    app.run(debug=True, port=5000)
