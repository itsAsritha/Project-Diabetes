
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained ML files
with open('imputer.pkl', 'rb') as file:
    imputer = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Read values from HTML
        num_preg = float(request.form['num_preg'])
        glucose_conc = float(request.form['glucose_conc'])
        diastolic_bp = float(request.form['diastolic_bp'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diab_pred = float(request.form['diab_pred'])
        age = float(request.form['age'])
        skin = float(request.form['skin'])

        # Arrange input features
        input_data = np.asarray([[
            num_preg,
            glucose_conc,
            diastolic_bp,
            insulin,
            bmi,
            diab_pred,
            age,
            skin
        ]])

        # Apply preprocessing
        imputed_data = imputer.transform(input_data)

        std_data = scaler.transform(imputed_data)

        # Predict
        prediction = rf_model.predict(std_data)

        if prediction[0] == 1:
            result = "Diabetes Detected"
        else:
            result = "No Diabetes Detected"

        return render_template(
            'index.html',
            prediction_text=result
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == '__main__':
    app.run(debug=True)
