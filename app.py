from flask import Flask, request, jsonify
import numpy as np
import pickle as pkl

# Load your trained model
with open('sales.pkl', 'rb') as f:
    model = pkl.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Sales Revenue Prediction API is up!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from JSON
        data = request.get_json(force=True)
        
        # Example: convert input list to numpy array
        features = np.array(data['features']).reshape(1, -1)

        # Predict using model
        prediction = model.predict(features)

        return jsonify({'predicted_sales_revenue': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
