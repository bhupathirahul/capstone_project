from flask import Flask, render_template, request, redirect, jsonify
import json
import os
import re
import numpy as np
import tensorflow as tf
import torch
from keras.models import load_model

app = Flask(__name__, static_url_path='/static')

# JSON file to store user data
USERS_FILE = 'users.json'

# Check if users.json exists, if not, create it
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f)

# Function to load users from JSON file
def load_users():
    with open(USERS_FILE) as f:
        return json.load(f)

# Function to save users to JSON file
def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for login
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    users = load_users()
    if username in users and users[username]['password'] == password:
        return render_template('predict.html')
    else:
        return render_template('index.html', error='Invalid username or password') 


# Function to validate password complexity
def is_valid_password(password):
    if len(password) < 8:
        return False
    if not re.search("[A-Z]", password):
        return False
    if not re.search("[0-9]", password):
        return False
    if not re.search("[!@#$%^&*?]", password):
        return False
    return True

# Route for signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')
    elif request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        error = None

        if not username.isalnum():
            error = 'Username must be alphanumeric!'
        elif not is_valid_password(password):
            error = 'Password must be at least 8 characters long and contain at least one uppercase letter, one number, and one special character!'
        elif username in users:
            error = 'Username already exists!'
        else:
            users[username] = {'password': password}
            save_users(users)
            return render_template('index.html', noterror='Registration successful!')

        return render_template('signup.html', error=error)


model = tf.keras.models.load_model('model_bnn.h5', compile=False)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    elif request.method == 'POST':
        # Get user input from the form
        precipitation = float(request.form['precipitation'])
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        wind = float(request.form['wind'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])

        # Make prediction
        user_input_array = np.array([precipitation, temp_max, temp_min, wind]).reshape(1, -1)
        output = model.predict(user_input_array)
        predicted_class = np.argmax(output, axis=1)[0]

        # Map predicted class to weather condition
        weather_mapping = {0: 'drizzle', 1: 'fog', 2: 'rain', 3: 'snow', 4: 'sun'}
        predicted_weather = weather_mapping.get(predicted_class, 'Unknown')

        return render_template('predict.html', predicted_weather=predicted_weather)



if __name__ == '__main__':
    app.run(debug=True)