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
		v1.GET("/2", aiethics.Lab2Code)
		v1.GET("/3", aiethics.Lab3Code)
		v1.GET("/4", aiethics.Lab4Code)
		v1.GET("/5", aiethics.Lab5Code)
		v1.GET("/6", aiethics.Lab1Code)
		v1.GET("/7", aiethics.Lab1Code)
		v1.GET("/8", aiethics.Lab1Code)
		v1.GET("/9", aiethics.Lab1Code)
		v1.GET("/10", aiethics.Lab1Code)
	}

	router.Run(":8080")
}
