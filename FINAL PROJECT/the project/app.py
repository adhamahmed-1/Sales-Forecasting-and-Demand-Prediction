from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def _parse_features(raw_features):
    """Convert a comma or space separated string into a list of floats."""
    if isinstance(raw_features, str):
        values = [item for item in re.split(r'[\s,]+', raw_features.strip()) if item]
    elif isinstance(raw_features, (list, tuple)):
        values = raw_features
    else:
        raise ValueError('Unsupported features format. Provide a list or comma separated string.')

    if not values:
        raise ValueError('Please provide at least one feature value.')

    try:
        return [float(v) for v in values]
    except ValueError as exc:
        raise ValueError('All feature values must be numeric.') from exc


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        if 'features' not in data:
            raise ValueError('JSON payload must include a "features" key.')

        feature_values = _parse_features(data['features'])

        # Convert to numpy array and reshape for the model
        features = np.array(feature_values).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)