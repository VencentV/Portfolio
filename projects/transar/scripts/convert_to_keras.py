import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import pickle

# Define the model architecture (This should match your original model's architecture)
def create_model():
    input_size = 40000  # 200x200 images flattened
    hidden_sizes = [512, 256, 128]
    output_size = 29

    model = Sequential([
        Dense(hidden_sizes[0], activation='relu', input_shape=(input_size,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(hidden_sizes[1], activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(hidden_sizes[2], activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(output_size, activation='softmax')
    ])
    return model

model = create_model()

# Function to load weights into the Keras model
def load_weights(model, pkl_filename):
    with open(pkl_filename, 'rb') as f:
        fcn = pickle.load(f)
    
    # Extract weights from FullyConnectedNet
    weights = []
    for i in range(1, fcn.num_layers + 1):
        weights.append(fcn.params['W%d' % i])
        weights.append(fcn.params['b%d' % i])
        if fcn.normalization and i < fcn.num_layers:
            weights.append(fcn.params['gamma%d' % i])
            weights.append(fcn.params['beta%d' % i])

    print(f"Extracted weights: {len(weights)} arrays")
    
    # Set weights in Keras model
    keras_weights = []
    layer_idx = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            keras_weights.append(weights[layer_idx])
            keras_weights.append(weights[layer_idx + 1])
            layer_idx += 2
        elif isinstance(layer, BatchNormalization):
            keras_weights.append(weights[layer_idx])
            keras_weights.append(weights[layer_idx + 1])
            keras_weights.append(np.zeros(weights[layer_idx + 1].shape))  # Moving mean
            keras_weights.append(np.ones(weights[layer_idx + 1].shape))  # Moving variance
            layer_idx += 2
        elif isinstance(layer, Dropout):
            continue  # Dropout layers do not have weights

    print(f"Keras model weights to be set: {len(keras_weights)} arrays")
    model.set_weights(keras_weights)

# Path to your .pkl file
pkl_filename = 'model/1000/trained_model (1000).pkl'
load_weights(model, pkl_filename)

# Save the Keras model in TensorFlow SavedModel format
model.save('model/1000/keras_model (1000)')

# You can run this command in your terminal or command prompt to convert to TensorFlow.js format:
# tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model WWW/model/keras_model WWW/model/tfjs_model
