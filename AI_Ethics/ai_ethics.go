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

func Lab1Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code1))
}
func Lab2Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code2))
}
func Lab3Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code3))
}
