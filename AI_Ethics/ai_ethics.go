package aiethics

import "github.com/gin-gonic/gin"

var code string = `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("Iris.csv")

print("First few rows of the dataset: \n", df.head())
        `

func Lab1Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code))
}
