package main

import (
	aai "github.com/SaiSawant1/exam/AAI"
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
		v1.GET("/6", aiethics.Lab6Code)
		v1.GET("/7", aiethics.Lab7Code)
		v1.GET("/8", aiethics.Lab8Code)
		v1.GET("/9", aiethics.Lab9Code)
		v1.GET("/10", aiethics.Lab10Code)
		v1.GET("/11", aiethics.Lab11Code)
	}

	v2 := router.Group("/aai")
	{
		v2.GET("/1", aai.Lab1Code)
		v2.GET("/2", aai.Lab2Code)
		v2.GET("/3", aai.Lab3Code)
		v2.GET("/4", aai.Lab4Code)
		v2.GET("/5", aai.Lab5Code)
		v2.GET("/6", aai.Lab6Code)
		v2.GET("/7", aai.Lab7Code)
		v2.GET("/8", aai.Lab8Code)
		v2.GET("/9", aai.Lab9Code)
		v2.GET("/10", aai.Lab10Code)
	}

	router.Run(":8080")
}
