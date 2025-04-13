from __future__ import print_function, division
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

import optim  # Updated import statement
from fc_net import FullyConnectedNet  # Importing the FullyConnectedNet
from solver import Solver  # Importing the Solver

import tensorflow as tf
import tensorflowjs as tfjs

# Define parameters
input_size = 40000  # 200x200 images flattened
hidden_sizes = [512, 256, 128]  # Increased number of hidden neurons and layers
output_size = 29  # 29 classes (A-Z, Space, Delete, Nothing)
std_dev = 1e-3  # Increased standard deviation for weight initialization
max_images_per_class = 2000  # To speed up loading/preprocessing
dropout_keep_prob = 0.5  # Dropout keep probability
learning_rate = 1e-3  # Adjust learning rate

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to {}".format(filename))

def load_and_preprocess_data(data_dir):
    print("Loading and preprocessing data...")
    images = []
    labels = []

    label_names = sorted(os.listdir(data_dir))
    label_map = {name: i for i, name in enumerate(label_names)}

    for label_name, label_idx in label_map.items():
        label_dir = os.path.join(data_dir, label_name)
        file_count = 0
        for filename in os.listdir(label_dir):
            if file_count >= max_images_per_class:
                break
            filepath = os.path.join(label_dir, filename)
            image = imread(filepath, as_gray=True)
            image = resize(image, (200, 200), anti_aliasing=True)
            images.append(image.flatten())
            labels.append(label_idx)
            file_count += 1
        print(f"Processed {file_count} images from class '{label_name}' (Label {label_idx}).")
    images = np.array(images)
    labels = np.array(labels)  # Keep labels as integers
    print("Data loaded and preprocessed. Total number of samples:", len(images))
    return images, labels

def accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == labels)

# Load data
print(f"Current working directory: {os.getcwd()}")
data_dir = 'data/asl_dataset/asl_alphabet_train'  
images, labels = load_and_preprocess_data(data_dir)

# Split data into train, validation, and test sets
print("Splitting data")
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the network
print("Initialized Network")
fcn = FullyConnectedNet(hidden_sizes, 
                        input_dim=input_size, 
                        num_classes=output_size, 
                        dropout_keep_ratio=dropout_keep_prob,
                        normalization='batchnorm', 
                        reg=0.0, 
                        weight_scale=std_dev, 
                        dtype=np.float32, 
                        seed=None)

# Train and evaluate the network
print("Training")
solver = Solver(fcn, 
                {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val},
                update_rule='adam', 
                optim_config={'learning_rate': learning_rate},
                lr_decay=0.95, 
                num_epochs=10, 
                batch_size=100, 
                print_every=20, 
                verbose=True)

# Start timer #
print("Starting Clock")
start_time = time.time()

solver.train()

# End timer #
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Calculate predictions
print("Calculating predictions")
y_train_pred = np.argmax(fcn.loss(X_train), axis=1)
y_val_pred = np.argmax(fcn.loss(X_val), axis=1)
y_test_pred = np.argmax(fcn.loss(X_test), axis=1)

# Calculate accuracies
train_acc = (y_train_pred == y_train).mean()
val_acc = (y_val_pred == y_val).mean()
test_acc = (y_test_pred == y_test).mean()

print('Training set accuracy: ', train_acc)
print('Validation set accuracy: ', val_acc)
print('Test set accuracy: ', test_acc)

# # Save the trained model
# model_dir = f'model/{max_images_per_class}/'
# os.makedirs(model_dir, exist_ok=True)
# save_model(fcn, f'{model_dir}trained_model.pkl')

# # Plotting
# print("Plotting")
# plt.figure()
# plt.plot(solver.train_acc_history, label='Train Accuracy')
# plt.plot(solver.val_acc_history, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.savefig(f'data/graphs/training_validation_accuracy ({max_images_per_class}).png')  

# plt.figure()
# plt.plot(solver.loss_history, label='Training Loss')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.legend()
# plt.savefig(f'data/graphs/training_loss ({max_images_per_class}).png') 
# plt.close()

# # Convert to Keras and TensorFlow.js
# print("Converting")

# def create_keras_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(hidden_sizes[0], activation='relu', input_shape=(input_size,)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(dropout_keep_prob),
#         tf.keras.layers.Dense(hidden_sizes[1], activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(dropout_keep_prob),
#         tf.keras.layers.Dense(hidden_sizes[2], activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(dropout_keep_prob),
#         tf.keras.layers.Dense(output_size, activation='softmax')
#     ])
#     return model

# # Function to load weights into the Keras model
# def load_weights_into_keras_model(model, fcn_model):
#     weights = []
#     for i in range(1, fcn_model.num_layers + 1):
#         weights.append(fcn_model.params['W%d' % i])
#         weights.append(fcn_model.params['b%d' % i])
#         if fcn_model.normalization and i < fcn_model.num_layers:
#             weights.append(fcn_model.params['gamma%d' % i])
#             weights.append(fcn_model.params['beta%d' % i])

#     keras_weights = []
#     layer_idx = 0
#     for layer in model.layers:
#         if isinstance(layer, tf.keras.layers.Dense):
#             keras_weights.append(weights[layer_idx])
#             keras_weights.append(weights[layer_idx + 1])
#             layer_idx += 2
#         elif isinstance(layer, tf.keras.layers.BatchNormalization):
#             keras_weights.append(weights[layer_idx])
#             keras_weights.append(weights[layer_idx + 1])
#             keras_weights.append(np.zeros(weights[layer_idx + 1].shape))  # Moving mean
#             keras_weights.append(np.ones(weights[layer_idx + 1].shape))  # Moving variance
#             layer_idx += 2
#         elif isinstance(layer, tf.keras.layers.Dropout):
#             continue  # Dropout layers do not have weights

#     model.set_weights(keras_weights)

# # Create and save Keras model
# keras_model = create_keras_model()
# load_weights_into_keras_model(keras_model, fcn)
# keras_model.save(f'{model_dir}keras_model')

# # Convert Keras model to TensorFlow.js format
# tfjs.converters.save_keras_model(keras_model, f'{model_dir}tfjs_model')
# print("Model conversion to Keras and TensorFlow.js completed.")
