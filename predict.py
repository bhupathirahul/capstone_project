from flask import Flask, render_template, request
import os
import torch
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('model_bnn.h5')

@app.route('/', methods=['GET', 'POST'])
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
