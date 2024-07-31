import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Initialize the network parameters
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 10

np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01

# Training the network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(x_train, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = softmax(output_layer_input)

    # Backward propagation
    output_error = y_train - output_layer_output
    output_delta = output_error

    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += np.dot(hidden_layer_output.T, output_delta) * learning_rate
    weights_input_hidden += np.dot(x_train.T, hidden_delta) * learning_rate

    # Print loss for every epoch
    loss = np.mean(np.abs(output_error))
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

# Testing the network
hidden_layer_input = np.dot(x_test, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
output_layer_output = softmax(output_layer_input)

# Calculate accuracy
predictions = np.argmax(output_layer_output, axis=1)
labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == labels)
print(f'Accuracy: {accuracy * 100}%')
