package main

import (
	aiethics "github.com/SaiSawant1/exam/AI_Ethics"
	"github.com/gin-gonic/gin"
)

func main() {

	router := gin.Default()

	v1 := router.Group("/ai_ethics")
	{
		v1.GET("/1", aiethics.Lab1Code)
	}

	router.Run(":8080")
}
