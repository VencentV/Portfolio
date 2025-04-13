from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load your model
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model('path/to/WWW/model/trained_model (100).pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Process your data (input formatting)
    input_data = np.array(data['input'])  # Adjust this based on how you're sending the data
    prediction = model.predict(input_data.reshape(1, -1))  # Reshape might be necessary
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # Runs on all interfaces on default port 5000
