package aip

import "github.com/gin-gonic/gin"

var code1 string = `
import numpy as np
import matplotlib.pyplot as plt

# Define the dataset
# Each point is represented as [x1, x2, label]
data = np.array([
    [2, 3, 1],
    [1, 1, 0],
    [2, 1, 1],
    [3, 3, 1],
    [1, 3, 0],
    [3, 2, 1],
    [1, 2, 0]
])

# Separate features and labels
X = data[:, :2]  # Features (x1, x2)
y = data[:, 2]   # Labels (0 or 1)

# Add a bias term (x0 = 1)
X = np.c_[np.ones(X.shape[0]), X]

# Perceptron parameters
weights = np.zeros(X.shape[1])  # Initialize weights to 0
learning_rate = 0.1
epochs = 10

# Perceptron training
for epoch in range(epochs):
    for i in range(len(X)):
        # Calculate the prediction (dot product)
        prediction = 1 if np.dot(weights, X[i]) > 0 else 0
        # Update weights if prediction is wrong
        weights += learning_rate * (y[i] - prediction) * X[i]

# Display the learned weights
print("Final weights:", weights)

# Plotting the dataset and decision boundary
def plot_decision_boundary(X, y, weights):
    plt.figure(figsize=(8, 6))
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i, 1], X[i, 2], color='blue', marker='o', label='Class 1' if i == 0 else "")
        else:
            plt.scatter(X[i, 1], X[i, 2], color='red', marker='x', label='Class 0' if i == 0 else "")

    # Decision boundary: w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0 + w1*x1) / w2
    x1_vals = np.linspace(0, 4, 100)
    x2_vals = -(weights[0] + weights[1] * x1_vals) / weights[2]
    plt.plot(x1_vals, x2_vals, color='green', label='Decision Boundary')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Perceptron Algorithm: Decision Boundary')
    plt.grid(True)
    plt.show()

plot_decision_boundary(X, y, weights)

`
var code2 string = `
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train / 255.0  # Scale pixel values to range [0, 1]
x_test = x_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)  # 10 classes (digits 0-9)
y_test = to_categorical(y_test, 10)

# Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),       # Flatten 28x28 images into 1D vectors
    Dense(128, activation='relu'),      # Hidden layer with 128 neurons
    Dense(10, activation='softmax')     # Output layer with 10 neurons for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
`
var code3 string = `
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train / 255.0  # Scale pixel values to range [0, 1]
x_test = x_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)  # 10 classes (digits 0-9)
y_test = to_categorical(y_test, 10)

# Function to build and train a model with a specified activation function
def build_and_train_model(activation):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return history, test_accuracy

# Train models with different activation functions
activations = ['relu', 'sigmoid', 'tanh']
histories = {}
test_accuracies = {}

for activation in activations:
    print(f"Training model with {activation} activation...")
    history, test_accuracy = build_and_train_model(activation)
    histories[activation] = history
    test_accuracies[activation] = test_accuracy

# Plot training and validation accuracy for each activation function
plt.figure(figsize=(12, 6))
for activation in activations:
    plt.plot(histories[activation].history['accuracy'], label=f'{activation.capitalize()} - Training')
    plt.plot(histories[activation].history['val_accuracy'], linestyle='dashed',
             label=f'{activation.capitalize()} - Validation')

plt.title('Training and Validation Accuracy for Different Activations')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Display test accuracies
print("\nTest Accuracies:")
for activation, accuracy in test_accuracies.items():
    print(f"{activation.capitalize()} Activation: {accuracy * 100:.2f}%")

`

var code4 string = `
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess data: Normalize images and convert labels to one-hot encoding
X_train = X_train / 255.0  # Scale pixel values to [0, 1]
X_test = X_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)  # One-hot encoding for 10 classes
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 output neurons for 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid()
plt.show()

`

var code5 string = `
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Generate a Fibonacci sequence
def generate_fibonacci(n):
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

# Prepare the dataset
def prepare_data(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:i + n_steps])
        y.append(sequence[i + n_steps])
    return np.array(X), np.array(y)

# Generate Fibonacci sequence
n_terms = 20  # Number of terms in the sequence
fibonacci_sequence = generate_fibonacci(n_terms)

# Normalize the sequence
fibonacci_sequence = np.array(fibonacci_sequence, dtype=np.float32)
fibonacci_sequence = (fibonacci_sequence - np.min(fibonacci_sequence)) / (np.max(fibonacci_sequence) - np.min(fibonacci_sequence))

# Prepare input-output pairs
n_steps = 3  # Number of steps to look back
X, y = prepare_data(fibonacci_sequence, n_steps)

# Reshape input to be [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)  # Output one prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X, y, epochs=200, verbose=0)

# Predict the next value in the sequence
new_input = fibonacci_sequence[-n_steps:]  # Last n_steps of the sequence
new_input = new_input.reshape((1, n_steps, 1))
predicted_value = model.predict(new_input, verbose=0)

print(f"Predicted next value (normalized): {predicted_value[0][0]}")

# Convert prediction back to original scale
predicted_value_original = predicted_value[0][0] * (np.max(fibonacci_sequence) - np.min(fibonacci_sequence)) + np.min(fibonacci_sequence)
print(f"Predicted next value (original scale): {predicted_value_original}")

# Plot the Fibonacci sequence and the prediction
plt.plot(range(len(fibonacci_sequence)), fibonacci_sequence, label='Original Sequence')
plt.scatter(len(fibonacci_sequence), predicted_value, color='red', label='Prediction')
plt.xlabel('Index')
plt.ylabel('Value (Normalized)')
plt.title('Fibonacci Sequence Prediction')
plt.legend()
plt.grid()
plt.show()

`
var code6 string = `
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import time

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define a simple neural network model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to train and time the model on a specific device
def train_on_device(device_name):
    with tf.device(device_name):
        model = create_model()

        start_time = time.time()
        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
        end_time = time.time()
        training_time = end_time - start_time
        _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        return training_time, test_accuracy

# Compare training times
print("Training on CPU...")
cpu_time, cpu_accuracy = train_on_device('/CPU:0')

gpu_time, gpu_accuracy = None, None
if tf.config.list_physical_devices('GPU'):
    print("Training on GPU...")
    gpu_time, gpu_accuracy = train_on_device('/GPU:0')
else:
    print("No GPU found.")

# Print results
print("\nResults:")
print(f"CPU - Training Time: {cpu_time:.2f} seconds, Test Accuracy: {cpu_accuracy:.2%}")
if gpu_time is not None:
    print(f"GPU - Training Time: {gpu_time:.2f} seconds, Test Accuracy: {gpu_accuracy:.2%}")

    # Speedup
    speedup = cpu_time / gpu_time
    print(f"GPU Speedup over CPU: {speedup:.2f}x")

`
var code7 string = `
import time
import numpy as np

# Simulate a small dataset
dataset_size = 10000
data = np.random.rand(dataset_size, 100)  # Example: 10,000 samples with 100 features

# Function to process data (replace with your actual processing logic)
def process_data(data):
    # Simulate a computation-intensive operation
    return np.sum(np.square(data), axis=1)

# CPU processing
start_time = time.time()
cpu_result = process_data(data)
cpu_time = time.time() - start_time
print(f"CPU processing time: {cpu_time:.4f} seconds")

# TPU processing (if available)
try:
    import tensorflow as tf
    # Check if TPU is available
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        tpu_start_time = time.time()
        tpu_data = tf.convert_to_tensor(data, dtype=tf.float32)

        # Perform the same computation on TPU
        tpu_result = np.sum(np.square(tpu_data.numpy()), axis=1)
        tpu_time = time.time() - tpu_start_time
        print(f"TPU processing time: {tpu_time:.4f} seconds")

        # Speedup comparison
        speedup = cpu_time / tpu_time
        print(f"TPU Speedup over CPU: {speedup:.2f}x")

except Exception as e:
    print(f"Error using TPU: {e}")
    print("TPU is not available or there’s an issue with the TPU setup.")

`
var code8 string = `
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
import time
import matplotlib.pyplot as plt

# Generate a synthetic classification dataset
def create_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                n_classes=2, random_state=42)
    return X.astype(np.float32), y

# Create a dataset
X, y = create_dataset()

# Split into training and testing sets
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Build a simple model
def create_model(optimizer):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model and time its performance
def train_and_time(optimizer, optimizer_name):
    start_time = time.time()
    model = create_model(optimizer)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    end_time = time.time()
    training_time = end_time - start_time
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return training_time, accuracy

# Train and compare using Adam and SGD optimizers
adam_time, adam_accuracy = train_and_time(tf.keras.optimizers.Adam(), "Adam")
sgd_time, sgd_accuracy = train_and_time(tf.keras.optimizers.SGD(), "SGD")

# Print results
print("Results:")
print(f"Adam - Training Time: {adam_time:.2f} seconds, Test Accuracy: {adam_accuracy:.2%}")
print(f"SGD - Training Time: {sgd_time:.2f} seconds, Test Accuracy: {sgd_accuracy:.2%}")

# Visualize the comparison
optimizers = ['Adam', 'SGD']
times = [adam_time, sgd_time]
accuracies = [adam_accuracy, sgd_accuracy]

fig, ax1 = plt.subplots()

# Bar plot for training times
color = 'tab:blue'
ax1.set_xlabel('Optimizer')
ax1.set_ylabel('Training Time (seconds)', color=color)
ax1.bar(optimizers, times, color=color, alpha=0.7, label='Training Time')
ax1.tick_params(axis='y', labelcolor=color)

# Line plot for accuracies
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(optimizers, accuracies, color=color, marker='o', linestyle='dashed', label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Comparison of Adam and SGD Optimizers')
plt.show()

`
var code9 = `
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the dataset using TensorFlow Datasets (TFDS)
def load_dataset():
    dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
    return dataset, info

# Preprocess the data
def preprocess(image, label):
    # Resize images to 150x150 and normalize pixel values to [0, 1]
    image = tf.image.resize(image, (150, 150))
    image = image / 255.0
    return image, label

# Load the dataset
dataset, info = load_dataset()

# Split the dataset into training and validation sets (80% train, 20% validation)
train_dataset = dataset['train']
total_size = info.splits['train'].num_examples
validation_size = int(0.2 * total_size)  # 20% for validation
train_dataset = train_dataset.skip(validation_size)
validation_dataset = dataset['train'].take(validation_size)

# Preprocess the datasets
train_dataset = (
    train_dataset.map(preprocess)
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
validation_dataset = (
    validation_dataset.map(preprocess)
    .batch(32)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset
)

# Evaluate the model on the validation dataset
loss, accuracy = model.evaluate(validation_dataset)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

`
var code10 string = `
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# Sample short text dataset
texts = [
    "I love programming in Python",
    "Python is great for data science",
    "Data science is fun and interesting",
    "I enjoy solving problems with Python",
    "Machine learning is a part of data science",
    "Deep learning is a subset of machine learning"
]

# Step 1: Text Preprocessing
# Tokenizing the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1  # Including 0 for padding

# Create sequences of words
input_sequences = []
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

# Padding sequences to ensure consistent input size
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Step 2: Prepare input and labels for training
X, y = input_sequences[:,:-1], input_sequences[:,-1]  # X contains the input, y contains the predicted word (next word)
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Step 3: Build the LSTM model
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_length-1),
    LSTM(150, return_sequences=False),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model
model.fit(X, y, epochs=1000, verbose=1)

# Step 5: Predict the next word
def predict_next_word(model, text, tokenizer):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')

    # Predict the next word
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted_probs)

    # Get the word from the tokenizer
    predicted_word = tokenizer.index_word[predicted_word_index]
    return predicted_word

# Test the model by predicting the next word
input_text = "I love programming"
predicted_word = predict_next_word(model, input_text, tokenizer)
print(f"Next word after '{input_text}': {predicted_word}")

`

func Lab1Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code1))
}
func Lab2Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code2))
}
func Lab3Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code3))
}
func Lab4Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code4))
}
func Lab5Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code5))
}
func Lab6Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code6))
}
func Lab7Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code7))
}
func Lab8Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code8))
}
func Lab9Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code9))
}
func Lab10Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code10))
}
