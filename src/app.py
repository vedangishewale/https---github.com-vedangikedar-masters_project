from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Allow CORS

# Load the scaler and KMeans model
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')

# Define cluster labels and renewal status
cluster_labels = {
    0: "High Score, Low Activity",
    1: "Moderate Score, Moderate Activity",
    2: "Low Score, High Activity",
    3: "Moderate Score, Moderate Balance",
    4: "High Score, High Balance",
    5: "High Score, High Tenure",
    6: "Low Score, Low Activity"
}

renewal_status = {
    0: "Renewed",
    1: "Renewed",
    2: "Not Renewed",
    3: "Not Renewed",
    4: "Renewed",
    5: "Renewed",
    6: "Not Renewed"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Extract features from request
    features = np.array([
        data['beaconScore'],
        data['mortgageBalance'],
        data['avgMonthlyTransactions'],
        data['visaBalance'],
        data['notMortgageBalance'],
        data['services'],
        data['tenureInMonths'],
        data['termInMonths'],
        data['termToMaturity'],
        data['interestRate']
    ]).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    cluster = kmeans.predict(features_scaled)[0]
    cluster_label = cluster_labels[cluster]
    renewal = renewal_status[cluster]
    
    # Return prediction as JSON
    return jsonify({
        'cluster': int(cluster),
        'cluster_label': cluster_label,
        'renewal_status': renewal
    })

if __name__ == '__main__':
    app.run(debug=True)
