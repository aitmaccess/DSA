import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input and output data
inputs = np.array([[0.66666667, 1.0],
                   [0.33333333, 0.55555556],
                   [1.0, 0.66666667]])

actual_output = np.array([[0.92], [0.86], [0.89]])

# Initialize weights randomly with mean 0
weights = np.random.rand(2, 1)

# Training the model (simple feedforward and backpropagation)
for _ in range(10000):
    input_layer = inputs
    predictions = sigmoid(np.dot(input_layer, weights))
    error = actual_output - predictions
    adjustments = error * sigmoid_derivative(predictions)
    weights += np.dot(input_layer.T, adjustments)

# Print results
print("Input:\n", inputs)
print("\nActual Output:\n", actual_output)
print("\nPredicted Output:\n", predictions)
