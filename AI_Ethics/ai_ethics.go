package aiethics

import "github.com/gin-gonic/gin"

var code1 string = `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("Iris.csv")

print("First few rows of the dataset: \n", df.head())
X = df['PetalLengthCm']
y = df['PetalWidthCm']

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Scatter Plot: Petal Length vs Petal Width')
plt.show()

correlation = np.corrcoef(X, y)[0,1]
print(f"Correlation coefficient between Petal Length and Petal Width: {correlation}")
model = LinearRegression()

X_reshaped = X.values.reshape(-1,1)

model.fit(X_reshaped, y)

y_pred= model.predict(X_reshaped)

plt.figure(figsize=(8,6))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X,y_pred,color='red', label='Regression Line')
plt.xlabel('Petal Length')
plt.ylabel("Prtal Width")
plt.title("Linear Regression: Petal Lenght vs Petal Width")
plt.legend()
plt.show()
print(f"Slope (Coefficient): {model.coef_[0]}")

print(f"Intercept: {model.intercept_}")
        `

var code2 string = `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('Iris.csv')

print("first few rows of the dataset:\n",df.head())

# Step 1: Preprocessing
# Remove rows where species is 'Iris-setosa'
df = df[df['Species'] != 'Iris-setosa']

# One-hot encoding of the 'species' column
df = pd.get_dummies(df, columns=['Species'], drop_first= True)
# Split data into features (X) and target (y)
X = df.drop('SepalLengthCm', axis=1)  # Features excluding target variable
y = df['SepalLengthCm']               # Target variable

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the model without a bias term
# Set fit_intercept=False to train without a bias term
model_no_bias = LinearRegression(fit_intercept=False)
model_no_bias.fit(X_train, y_train)

# Step 3: Train the model with a bias term
model_with_bias = LinearRegression(fit_intercept=True)
model_with_bias.fit(X_train, y_train)

# Step 4: Model evaluation
# Predictions
y_pred_no_bias = model_no_bias.predict(X_test)
y_pred_with_bias = model_with_bias.predict(X_test)

# Calculate Mean Squared Error for both models
mse_no_bias = mean_squared_error(y_test, y_pred_no_bias)
mse_with_bias = mean_squared_error(y_test, y_pred_with_bias)

print(f"Mean Squared Error without bias: {mse_no_bias}")
print(f"Mean Squared Error with bias: {mse_with_bias}")

# Step 5: Plotting the results
# Plot predictions vs actual values for both models
plt.figure(figsize=(10, 5))

# Plot for model without bias
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_no_bias, color='blue')
plt.title('Model Without Bias')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

# Plot for model with bias
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_with_bias, color='green')
plt.title('Model With Bias')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

plt.tight_layout()
plt.show()

`

var code3 string = `
import gensim.downloader as api
import numpy as np
model = api.load('word2vec-google-news-300')
#Words for analogy
word_a, word_b, word_c = 'delhi' , 'india' , 'colombo'
#find the words that completes
result = model.most_similar(positive=[word_b, word_c], negative=[word_a], topn=1)
word_d = result[0][0]
print(f"{word_a} is to {word_b} as {word_c} is to {word_d} ")
print(f"word: {word_d}, Similarity: {result[0][1]}")
words = [word_a, word_b, word_c, word_d]
word_vectors = np.array([model[word] for word in words])

for i, word in enumerate(words):
  print(f"Vector for '{word}': {word_vectors[i]}")

# Assuming 'model' is already defined and loaded with word vectors

paragraph = """ AI ethics are the set of guiding principles that stakeholders (from engineers to government officials) use to ensure artificial intelligence technology is developed and used responsibly. This means taking a safe, secure, humane, and environmentally friendly approach to AI."""

words = paragraph.lower().split()

key_terms = ["ethics", "intelligence", "humane"]

for term in key_terms:
    if term in words:
        similar_words = model.most_similar(term, topn=5)
        print(f"Words similar to '{term}':")
        for word, similarity in similar_words:
            print(f"({word}): ({similarity})")
        print("\n")
    else:
        print(f"'{term}' not found in the paragraph.\n")

`

var code4 string = `
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
# Load the dataset from UCI repository (example with Iris dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, names=column_names)
# Extracting features and target variable
X = data.drop('species', axis=1)
y = data['species']
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
learning_rate = 0.01
iteration = 1000
# Fitting a perceptron model without bias
model_no_bias = Perceptron(fit_intercept=False, eta0 = learning_rate, max_iter = iteration)
model_no_bias.fit(X_train, y_train)
y_pred_no_bias = model_no_bias.predict(X_test)
accuracy_no_bias = accuracy_score(y_test, y_pred_no_bias)
# Fitting a perceptron model with bias
model_with_bias = Perceptron(fit_intercept=True, eta0 = learning_rate, max_iter = iteration)
model_with_bias.fit(X_train, y_train)
y_pred_with_bias = model_with_bias.predict(X_test)
accuracy_with_bias = accuracy_score(y_test, y_pred_with_bias)
print("Accuracy of perceptron without bias:", accuracy_no_bias)
print("Accuracy of perceptron with bias:", accuracy_with_bias)
`
var code5 string = `
import pandas as pd

def check_class_imbalance(labels):
  class_0_count = (labels == 0).sum()
  class_1_count = (labels == 1).sum()

  print(f"Class 0 count: {class_0_count}")
  print(f"Class 1 count: {class_1_count}")

if __name__ == "__main__":
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
  column_names = ["variance", "skewness", "curtosis", "entropy", "class"]
  data = pd.read_csv(url, header=None, names=column_names)

  print("Banknote Authentication Dataset:")
  print(data)

  labels = data["class"]

  print("\nExtracted Labels: ", labels.values)

  check_class_imbalance(labels)
`

var code6 string = `
import pandas as pd

# URL for the UCI Adult Income dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

# Column names for the dataset
column_names = [
    'Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num',
    'Marital-Status', 'Occupation', 'Relationship', 'Race',
    'Sex', 'Capital-Gain', 'Capital-Loss', 'Hours-per-week',
    'Native-Country', 'Income'
]

# Load the dataset
data = pd.read_csv(url, header=None, names=column_names)

# Function to anonymize the dataset
def anonymize_dataset(df):
    # Anonymize specific columns as needed
    if 'Age' in df.columns:
        df['Age'] = 'REDACTED'  # You could also anonymize it differently

    if 'Workclass' in df.columns:
        df['Workclass'] = 'REDACTED'

    if 'Occupation' in df.columns:
        df['Occupation'] = 'REDACTED'

    if 'Native-Country' in df.columns:
        df['Native-Country'] = 'REDACTED'

    # Income column (target variable) can also be anonymized
    if 'Income' in df.columns:
        df['Income'] = df['Income'].replace({'<=50K': 'REDACTED', '>50K': 'REDACTED'})

    return df

# Anonymize the dataset
anonymized_df = anonymize_dataset(data)

# Display the original and anonymized datasets
print("Original Dataset:\n", data.head())
print("\nAnonymized Dataset:\n", anonymized_df.head())

import pandas as pd

# URL for the UCI Adult Income dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

# Column names for the dataset
column_names = [
    'Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num',
    'Marital-Status', 'Occupation', 'Relationship', 'Race',
    'Sex', 'Capital-Gain', 'Capital-Loss', 'Hours-per-week',
    'Native-Country', 'Income'
]

# Load the dataset
data = pd.read_csv(url, header=None, names=column_names)

# Function to apply Caesar cipher
def caesar_cipher(text, shift):
    result = []
    for char in text:
        if char.isalpha():
            # Shift within the alphabet
            shift_base = ord('A') if char.isupper() else ord('a')
            result.append(chr((ord(char) - shift_base + shift) % 26 + shift_base))
        else:
            result.append(char)  # Non-alphabetical characters remain unchanged
    return ''.join(result)

# Function to anonymize the dataset
def anonymize_dataset(df, shift):
    # Anonymize specific columns using Caesar cipher
    if 'Workclass' in df.columns:
        df['Workclass'] = df['Workclass'].apply(lambda x: caesar_cipher(x, shift))

    if 'Occupation' in df.columns:
        df['Occupation'] = df['Occupation'].apply(lambda x: caesar_cipher(x, shift))

    if 'Native-Country' in df.columns:
        df['Native-Country'] = df['Native-Country'].apply(lambda x: caesar_cipher(x, shift))

    # Income column (target variable) can also be anonymized
    if 'Income' in df.columns:
        df['Income'] = df['Income'].replace({'<=50K': 'REDACTED', '>50K': 'REDACTED'})

    return df

# Anonymize the dataset with a Caesar cipher shift of 3
shift_value = 3
anonymized_df = anonymize_dataset(data, shift_value)

# Display the original and anonymized datasets
print("Original Dataset:\n", data.head())
print("\nAnonymized Dataset:\n", anonymized_df.head())

import pandas as pd

# Sample dataset with PII (Replace this with your dataset or use an external dataset)
data = {
    'Name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'David Lee'],
    'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'david@example.com'],
    'Age': [28, 34, 22, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
    'Phone': ['123-456-7890', '098-765-4321', '555-555-5555', '999-999-9999']
}

# Load the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Function to anonymize the dataset
def anonymize_dataset(df, pii_columns, method="mask"):
    """
    Anonymizes a dataset by removing or masking personally identifiable information (PII).

    Parameters:
    df (pd.DataFrame): The input dataset.
    pii_columns (list): List of columns containing PII to anonymize.
    method (str): The anonymization method ("remove" or "mask").

    Returns:
    pd.DataFrame: Anonymized dataset.
    """
    anonymized_df = df.copy()  # Create a copy to avoid modifying the original dataset

    if method == "remove":
        # Drop PII columns
        anonymized_df = anonymized_df.drop(columns=pii_columns)
    elif method == "mask":
        # Mask PII values with anonymized text
        for column in pii_columns:
            anonymized_df[column] = anonymized_df[column].apply(lambda _: f"{column}123")

    return anonymized_df

# Define PII columns
pii_columns = ['Name', 'Email', 'Phone']

# Test the function with both anonymization methods
anonymized_df_remove = anonymize_dataset(df, pii_columns, method="remove")
anonymized_df_mask = anonymize_dataset(df, pii_columns, method="mask")

# Display original and anonymized datasets
print("Original Dataset:\n", df)
print("\nAnonymized Dataset (Remove PII):\n", anonymized_df_remove)
print("\nAnonymized Dataset (Mask PII):\n", anonymized_df_mask)

`

var code7 string = `
import pandas as pd

def detect_anomalies(logs, threshold):
  anomalies = logs[logs['error_rate'] > threshold]
  return anomalies

#Example usage
if __name__ == "__main__":
  # Load logs from a CSV file (replace the file name with your dataset)
  logs = pd.read_csv('/content/sample_logs_1.csv')

#Display the first few reows of the dataset to understand its structure
print(logs.head())

#specify the threshold
threshold = 0.19 # Adjust this based on the dataset's context
anomalies = detect_anomalies(logs, threshold)

print("Anomalies detected:")
print(anomalies)
`

var code8 string = `
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def calculate_sma(data, window):
    """
    Calculate Simple Moving Average

    Parameters:
    data (list): List of values to calculate SMA
    window (int): Window size for moving average

    Returns:
    list: Simple moving averages
    """
    sma = []
    for i in range(len(data)):
        if i < window - 1:
            sma.append(np.nan)
        else:
            sma.append(np.mean(data[i-(window-1):i+1]))
    return sma

def monitor_performance(metrics, sma_window=5):
    """
    Monitor and plot performance metrics

    Parameters:
    metrics (dict): Dictionary containing training metrics
    sma_window (int): Window size for moving average
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot loss
    ax1.plot(metrics['loss'], label='Training Loss', alpha=0.5)
    ax1.plot(calculate_sma(metrics['loss'], sma_window),
             label=f'SMA-{sma_window} Loss', linewidth=2)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(metrics['accuracy'], label='Training Accuracy', alpha=0.5)
    ax2.plot(metrics['val_accuracy'], label='Validation Accuracy', alpha=0.5)
    ax2.plot(calculate_sma(metrics['accuracy'], sma_window),
             label=f'SMA-{sma_window} Train Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy Over Time')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def train_model(X_train, X_val, y_train, y_val, max_epochs=100):
    """
    Train neural network on Iris dataset

    Parameters:
    X_train, X_val: Training and validation features
    y_train, y_val: Training and validation labels
    max_epochs: Maximum number of training epochs

    Returns:
    dict: Dictionary containing training metrics
    """
    # Initialize model
    model = MLPClassifier(
        hidden_layer_sizes=(10, 5),
        max_iter=1,
        warm_start=True,  # Allows incremental training
        random_state=42
    )

    # Initialize metrics dictionary
    metrics = {
        'loss': [],
        'accuracy': [],
        'val_accuracy': [],
        'timestamp': []
    }

    print("Starting training...")
    for epoch in range(max_epochs):
        # Train for one epoch
        model.fit(X_train, y_train)

        # Record metrics
        metrics['loss'].append(model.loss_)
        metrics['accuracy'].append(accuracy_score(y_train, model.predict(X_train)))
        metrics['val_accuracy'].append(accuracy_score(y_val, model.predict(X_val)))
        metrics['timestamp'].append(datetime.now())

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}")
            print(f"Loss: {metrics['loss'][-1]:.4f}")
            print(f"Train Accuracy: {metrics['accuracy'][-1]:.4f}")
            print(f"Validation Accuracy: {metrics['val_accuracy'][-1]:.4f}")
            print("-" * 40)

    return metrics, model

def main():
    """
    Main function to load data, train model, and visualize results
    """
    # Load Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train model and get metrics
    metrics, model = train_model(X_train, X_val, y_train, y_val, max_epochs=100)

    # Calculate moving averages
    sma_window = 5
    loss_sma = calculate_sma(metrics['loss'], sma_window)
    accuracy_sma = calculate_sma(metrics['accuracy'], sma_window)

    # Create DataFrame for easy analysis
    df = pd.DataFrame({
        'Loss': metrics['loss'],
        'Loss_SMA': loss_sma,
        'Train_Accuracy': metrics['accuracy'],
        'Val_Accuracy': metrics['val_accuracy'],
        'Accuracy_SMA': accuracy_sma,
        'Timestamp': metrics['timestamp']
    })

    # Display summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Monitor and plot performance
    print("\nGenerating performance plots...")
    monitor_performance(metrics, sma_window=5)

    # Print final model performance
    print("\nFinal Model Performance:")
    print(f"Training Accuracy: {metrics['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {metrics['val_accuracy'][-1]:.4f}")

    return df, model

if __name__ == "__main__":
    df, model = main()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def simple_moving_average(data, window_size):
    """
    Calculate the Simple Moving Average (SMA).

    Parameters:
    - data: list of numeric values
    - window_size: size of the moving window

    Returns:
    - sma: list of smoothed values
    """
    sma = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return sma

def monitor_performance(X_train, y_train, X_test, y_test, epochs, window_size):
    """
    Monitor and plot performance metrics (accuracy and error rate) of a model over epochs.

    Parameters:
    - X_train, y_train: training data and labels
    - X_test, y_test: testing data and labels
    - epochs: number of training epochs
    - window_size: window size for the SMA
    """
    model = LogisticRegression(max_iter=1, solver='saga', warm_start=True)

    accuracy_history = []
    error_rate_history = []

    for epoch in range(epochs):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy

        accuracy_history.append(accuracy)
        error_rate_history.append(error_rate)
        print(f'Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.4f}, Error Rate: {error_rate:.4f}')

    # Calculate SMAs
    smoothed_accuracy = simple_moving_average(accuracy_history, window_size)
    smoothed_error_rate = simple_moving_average(error_rate_history, window_size)

    # Plotting
    plt.figure(figsize=(14, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, label='Accuracy', color='blue')
    plt.plot(range(window_size, len(smoothed_accuracy) + window_size), smoothed_accuracy, label='SMA Accuracy', color='orange')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Error Rate plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(error_rate_history) + 1), error_rate_history, label='Error Rate', color='red')
    plt.plot(range(window_size, len(smoothed_error_rate) + window_size), smoothed_error_rate, label='SMA Error Rate', color='purple')
    plt.title('Model Error Rate Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Simulate Data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Monitor performance
monitor_performance(X_train, y_train, X_test, y_test, epochs=50, window_size=5)

`
var code9 = `
import logging
import random
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename='system_log.txt',
    filemode='a'  # Append to the log file
)

# Define AISystem class
class AISystem:
    def __init__(self):
        # Initialize load with a default value
        self.load = 0.0

    def update_load(self):
        # Simulate load with a random value between 0 and 1
        self.load = random.uniform(0, 1)
        return self.load

# Define the monitoring function
def monitor_system(ai_system, threshold=0.8):
    # Update and get the current load
    current_load = ai_system.update_load()
    if current_load >= threshold:
        # Trigger fail-safe if load exceeds the threshold
        message = f"ALERT: Fail-safe triggered! Current load: {current_load:.2f} exceeds threshold: {threshold:.2f}"
        logging.info(message)
        print(message)
    else:
        # Log normal operation status
        message = f"System operating normally. Current load: {current_load:.2f}"
        logging.info(message)
        print(message)

# Main script
if __name__ == "__main__":
    # Instantiate AISystem
    ai_system = AISystem()

    # Run monitoring in a loop
    print("Starting system monitoring...")
    try:
        for _ in range(10):  # Example: run monitoring 10 times
            monitor_system(ai_system, threshold=0.8)
            time.sleep(2)  # Delay for 2 seconds between checks
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")

`
var code10 = `
!pip install fairlearn scikit-learn numpy

# Step 1: Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate

# Step 2: Create a synthetic dataset with bias
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    random_state=42
)
sensitive_feature = np.random.choice(["Male", "Female"], size=1000, p=[0.6, 0.4])

# Step 3: Split the dataset into training and test sets
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)

# Step 4: Train an unconstrained (original) model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Original Model Accuracy: {accuracy:.2f}")

# Step 5: Calculate fairness metrics
metric_frame = MetricFrame(
    metrics=selection_rate,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_test
)
print("Original Selection Rates by Group:")
print(metric_frame.by_group)

# Step 6: Apply ExponentiatedGradient with DemographicParity
fair_model = LogisticRegression(solver="liblinear")
constraint = DemographicParity()
mitigator = ExponentiatedGradient(estimator=fair_model, constraints=constraint)

# Train the fair model
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
y_pred_fair = mitigator.predict(X_test)

# Step 7: Evaluate the fair model
accuracy_fair = accuracy_score(y_test, y_pred_fair)
print(f"Fair Model Accuracy: {accuracy_fair:.2f}")

# Step 8: Recalculate fairness metrics
metric_frame_fair = MetricFrame(
    metrics=selection_rate,
    y_true=y_test,
    y_pred=y_pred_fair,
    sensitive_features=sensitive_test
)
print("Fair Model Selection Rates by Group:")
print(metric_frame_fair.by_group)

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
