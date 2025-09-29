from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('Models/ridge.pk1','rb'))
standard_scalar = pickle.load(open('Models/scalar.pk1','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'POST':
        # Collect input from form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Arrange data into array
        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Apply scaling
        scaled_data = standard_scalar.transform(input_data)

        # Make prediction
        prediction = ridge_model.predict(scaled_data)[0]

        # Send prediction back to template
        return render_template('home.html', result=round(prediction, 2))
    else:
        
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
