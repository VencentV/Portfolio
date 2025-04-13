import pickle
import tensorflow as tf
from tensorflow import keras

# Load your existing model from a pickle file
def load_model_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Rebuild the loaded model in Keras
def rebuild_keras_model(loaded_model):
    # Example of rebuilding a simple dense network, adjust according to your model's architecture
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(40000,)),  # Adjust input_shape based on your model
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(29, activation='softmax')  # Assuming 29 output classes
    ])
    
    # Compile model (adjust parameters as necessary)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Path to your .pkl file
pickle_file_path = 'model/trained_model (500).pkl'

# Load and rebuild the model
loaded_model = load_model_from_pickle(pickle_file_path)
keras_model = rebuild_keras_model(loaded_model)

# Save the rebuilt model
keras_model.save('model/keras_model')
