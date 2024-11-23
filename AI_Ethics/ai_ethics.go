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

func Lab1Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code1))
}
