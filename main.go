package main

import "github.com/gin-gonic/gin"

func main() {

	router := gin.Default()

	v1 := router.Group("/ai_ethics")
	{
		v1.GET("/1", func(c *gin.Context) {
			c.JSON(200, gin.H{
				"message": "pong",
			})
		})
	}

	router.Run(":8080")
}
