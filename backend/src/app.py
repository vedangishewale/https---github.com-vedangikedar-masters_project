# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import joblib

# Load the model, scaler, and PCA
model_interest_rate = pickle.load(open('linear_regression_model.pkl', 'rb'))
model_customer_segmentation = joblib.load('kmeans_model.pkl')
model_scalar = joblib.load('scaler.pkl')

# Define cluster labels based on characteristics
cluster_labels = {
    0: "High Score, Low Activity",
    1: "Moderate Score, Moderate Activity",
    2: "Low Score, High Activity",
    3: "Moderate Score, Moderate Balance",
    4: "High Score, High Balance",
    5: "High Score, High Tenure",
    6: "Low Score, Low Activity"
}

# Define renewal status based on cluster
renewal_status = {
    0: "Renewed",
    1: "Renewed",
    2: "Not Renewed",
    3: "Not Renewed",
    4: "Renewed",
    5: "Renewed",
    6: "Not Renewed"
}


app = Flask(__name__)
CORS(app)

@app.route('/predictinterestrate', methods=['POST'])
def predictinterestrate():
    try:
        data = request.get_json()
        features = ['Beacon_Score', 'Services', 'Avg_Monthly_Transactions', 'Has_Payroll',
                    'Has_Investment', 'Has_Visa', 'Age', 'Tenure_In_Months',
                    'TermToMaturity', 'NumberOfParties']
        
        # Ensure all required features are present
        if not all(feature in data for feature in features):
            return jsonify({"error": "Missing feature(s)"}), 400
        
        # Extract and convert features from the request
        input_data = []
        for feature in features:
            try:
                input_data.append(data[feature])
            except ValueError:
                return jsonify({"error": f"Invalid value for {feature}, must be numeric"}), 400
        
        input_data = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = abs(model_interest_rate.predict(input_data))
        
        return jsonify({"interest_rate": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/prediccustomersegmentation', methods=['POST'])
def prediccustomersegmentation():
    try:
        data = request.get_json()
        features = ['Beacon_Score', 'Mortgage_Balance', 'Services', 'Avg_Monthly_Transactions', 
            'Has_Payroll', 'Has_Investment', 'Has_Visa', 'VISA_balance', 'Has_Deposit', 
            'not_mortgage_lending', 'Tenure_In_Months', 'TermInMonths', 
            'TermToMaturity', 'InterestRate']
        
        # Ensure all required features are present
        if not all(feature in data for feature in features):
            return jsonify({"error": "Missing feature(s)"}), 400
        
        # Extract and convert features from the request
        input_data = []
        for feature in features:
            try:
                input_data.append(data[feature])
            except ValueError:
                return jsonify({"error": f"Invalid value for {feature}, must be numeric"}), 400

        input_data = np.array(input_data).reshape(1, -1)

        print(input_data)

        try:
            input_data_scaled = model_scalar.transform(input_data)
            print(f"Scaled data: {input_data_scaled}")
        except Exception as e:
            print(f"Error during scaling: {str(e)}")
            return jsonify({"error": f"Error during scaling: {str(e)}"}), 500
        


        try:
            prediction = model_customer_segmentation.predict(input_data_scaled)
            print(f"Scaled data: {prediction}")
        except Exception as e:
            print(f"Error during scaling: {str(e)}")
            return jsonify({"error": f"Error during scaling: {str(e)}"}), 500

        label = cluster_labels.get(prediction[0])
        renewal = renewal_status.get(prediction[0])

        return jsonify({"label": label, "renewal": renewal})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
