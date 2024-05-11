from flask import Flask, render_template, request, redirect, jsonify
import json
import os
import re
import numpy as np
import tensorflow as tf
import torch
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import requests
import csv

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

def fetch_weather_and_generate_graphs(location):
    api_key = "3c813e351d3d3e42e2d10d9c8bac9973"
    full_link = "https://api.openweathermap.org/data/2.5/forecast?q=" + location + "&appid=" + api_key
    api_link = requests.get(full_link)
    api_data = api_link.json()

    # Check if the API response contains data for the provided location
    if 'list' not in api_data:
        return False  # Return False if the location is invalid or no data is available

    rows = []
    max_temperatures = {}
    min_temperatures = {}

    for item in api_data['list']:
        date = item['dt_txt'].split(' ')[0]
        temp = int(item['main']['temp'] - 273.15)
        temp_min = int(item['main']['temp_min'] - 273.15)
        temp_max = int(item['main']['temp_max'] - 273.15)
        humidity = item['main']['humidity']
        speed = int(item['wind']['speed'])
        rows.append([date, temp, temp_min, temp_max, humidity, speed])
        
        if date not in max_temperatures:
            max_temperatures[date] = temp_max
        else:
            max_temperatures[date] = max(max_temperatures[date], temp_max)
            
        if date not in min_temperatures:
            min_temperatures[date] = temp_min
        else:
            min_temperatures[date] = min(min_temperatures[date], temp_min)

    # Write data to CSV file
    with open('weather_data.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Date', 'Temperature (C)', 'Min Temperature (C)', 'Max Temperature (C)', 'Humidity', 'Speed'])
        writer.writerows(rows)

    # Read data from CSV file using pandas
    df = pd.read_csv('weather_data.csv')

    # Group by date and find max and min temperatures
    Temperature_grouped = df.groupby('Date').agg({'Max Temperature (C)': 'max', 'Min Temperature (C)': 'min'})
    Humidity_grouped = df.groupby('Date').agg({'Humidity': 'max'})

    # Plotting graphs
    plt.figure(figsize=(8, 2))
    Temperature_grouped.plot(marker='o')
    plt.xlabel('Date')
    plt.ylabel('Temperature (C)')
    plt.title('Max and Min Temperatures in ' + location)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(['Max Temperature (C)', 'Min Temperature (C)'])
    plt.savefig('static/temperature_graph.png')  # Save the temperature graph as an image

    plt.figure(figsize=(8, 2))
    Humidity_grouped.plot(marker='o')
    plt.title('Humidity in ' + location)
    plt.xlabel('Date')
    plt.ylabel('Humidity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/humidity_graph.png')  # Save the humidity graph as an image

    # Clear the CSV file
    open('weather_data.csv', 'w').close()

    return True  # Return True if data was fetched and graphs were generated successfully

# Route for location page
@app.route('/location', methods=['POST'])
def location():
    location = request.form['location']
    if fetch_weather_and_generate_graphs(location):
        return render_template('location.html', location=location)
    else:
        return render_template('predict.html', message='Invalid location. Please provide a valid location.')

if __name__ == '__main__':
    app.run(debug=True)