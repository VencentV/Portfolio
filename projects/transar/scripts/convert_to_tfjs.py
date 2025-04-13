import tensorflowjs as tfjs
import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('model/1000/keras_model (1000)')

# Convert the model to TensorFlow.js format
tfjs.converters.save_keras_model(model, 'model/1000/tfjs_model (1000)')
print("fin")