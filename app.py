# app.py
from flask import Flask, render_template, request, flash
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flashing messages

# Load the model when the application starts
def load_model(model_path='fraud_detection_model.pkl'):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise Exception(f"Model file {model_path} not found. Please train and save the model first.")

# Load model components
try:
    model_components = load_model()
    MODEL = model_components['model']
    SCALER = model_components['scaler']
    ENCODER = model_components['encoder']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    MODEL = None
    SCALER = None
    ENCODER = None

def predict_fraud(data):
    """Make fraud prediction on input data"""
    try:
        # Create DataFrame with the input data
        test_data = pd.DataFrame({
            'trans_date_trans_time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'cc_num': [data['cc_num']],
            'merchant': [data['merchant']],
            'category': [data['category']],
            'amt': [float(data['amt'])],
            'first': [data['first']],
            'last': [data['last']],
            'gender': [data['gender']],
            'street': [data['street']],
            'city': [data['city']],
            'state': [data['state']],
            'zip': [int(data['zip'])],
            'lat': [float(data['lat'])],
            'long': [float(data['long'])],
            'city_pop': [int(data['city_pop'])],
            'merch_lat': [float(data['merch_lat'])],
            'merch_long': [float(data['merch_long'])]
        })

        # Feature engineering
        test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time'])
        test_data['trans_day'] = test_data['trans_date_trans_time'].dt.day
        test_data['trans_month'] = test_data['trans_date_trans_time'].dt.month
        test_data['trans_year'] = test_data['trans_date_trans_time'].dt.year
        test_data['trans_hour'] = test_data['trans_date_trans_time'].dt.hour
        test_data['trans_minute'] = test_data['trans_date_trans_time'].dt.minute
        test_data['weekday'] = test_data['trans_date_trans_time'].dt.weekday
        test_data['is_weekend'] = test_data['weekday'].isin([5, 6]).astype(int)

        # Calculate distance
        test_data['distance'] = ((test_data['lat'] - test_data['merch_lat'])**2 +
                                 (test_data['long'] - test_data['merch_long'])**2)**0.5

        # Handle categorical variables
        try:
            test_data['category'] = ENCODER.transform(test_data['category'])
        except ValueError:
            test_data['category'] = -1

        try:
            test_data['cc_num'] = ENCODER.transform(test_data['cc_num'])
        except ValueError:
            test_data['cc_num'] = -1

        # Scale numerical features
        numerical_features = ['amt', 'zip', 'city_pop', 'distance']
        test_data[numerical_features] = SCALER.transform(test_data[numerical_features])

        # Ensure correct column order
        expected_columns = ['cc_num', 'category', 'amt', 'zip', 'lat', 'long',
                            'city_pop', 'merch_lat', 'merch_long', 'trans_day',
                            'trans_month', 'trans_year', 'trans_hour', 'trans_minute',
                            'weekday', 'is_weekend', 'distance']
        test_data = test_data[expected_columns]

        # Make prediction
        prediction = MODEL.predict(test_data)[0]
        prediction_prob = MODEL.predict_proba(test_data)[0]

        return {
            'prediction': int(prediction),
            'probability': float(prediction_prob[1]),
            'status': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'confidence': f"{max(prediction_prob):.2%}"
        }

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'cc_num': request.form['cc_num'],
                'merchant': request.form['merchant'],
                'category': request.form['category'],
                'amt': request.form['amt'],
                'first': request.form['first'],
                'last': request.form['last'],
                'gender': request.form['gender'],
                'street': request.form['street'],
                'city': request.form['city'],
                'state': request.form['state'],
                'zip': request.form['zip'],
                'lat': request.form['lat'],
                'long': request.form['long'],
                'city_pop': request.form['city_pop'],
                'merch_lat': request.form['merch_lat'],
                'merch_long': request.form['merch_long']
            }

            prediction_result = predict_fraud(form_data)
            if prediction_result is None:
                flash('Error making prediction. Please check your input data.', 'error')

        except Exception as e:
            flash(f'Error: {str(e)}', 'error')

    return render_template('index.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)