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
