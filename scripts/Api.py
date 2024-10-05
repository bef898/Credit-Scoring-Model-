from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\befekadum\Documents\10x acadamy\week6\Credit-Scoring-Model-\notebooks\log_reg.pkl')  

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Convert the input data to a DataFrame (modify this as per your input structure)
    input_data = pd.DataFrame(data, index=[0])

    # Preprocess input data if needed (e.g., binning, scaling)
    # Example: input_data['Recency_bin'] = pd.cut(input_data['Recency'], bins=5)

    # Make predictions
    predictions = model.predict(input_data)

    # Return the predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
