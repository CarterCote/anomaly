from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib

# Load the trained models
iso_forest = joblib.load('iso_forest_model.pkl')
rf = joblib.load('rf_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)

    # Preprocess the data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df = pd.get_dummies(df, columns=['event_type', 'metadata'], drop_first=True)

    scaler = StandardScaler()
    numerical_features = ['transaction_amount', 'threshold']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    X = df.drop(['timestamp', 'user_id', 'account_id'], axis=1)

    # Predict with Isolation Forest
    df['anomaly_score'] = iso_forest.decision_function(X)

    # Predict with Random Forest
    predictions = rf.predict(X)

    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
