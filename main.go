package main

import "github.com/gin-gonic/gin"

func main() {

	router := gin.Default()

	v1 := router.Group("/ai_ethics")
	{
		v1.GET("/1", func(c *gin.Context) {
			c.JSON(200, gin.H{
				"code": `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("Iris.csv")

print("First few rows of the dataset: \n", df.head())
        `,
			})
		})
	}

	router.Run(":8080")
}
