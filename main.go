package main

import (
	aai "github.com/SaiSawant1/exam/AAI"
	aip "github.com/SaiSawant1/exam/AIP"
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
		v2.GET("/11", aai.Lab11Code)
		v2.GET("/12", aai.Lab12Code)
	}
	v3 := router.Group("/aip")
	{
		v3.GET("/1", aip.Lab1Code)
		v3.GET("/2", aip.Lab2Code)
		v3.GET("/3", aip.Lab3Code)
		v3.GET("/4", aip.Lab4Code)
		v3.GET("/5", aip.Lab5Code)
		v3.GET("/6", aip.Lab6Code)
		v3.GET("/7", aip.Lab7Code)
		v3.GET("/8", aip.Lab8Code)
		v3.GET("/9", aip.Lab9Code)
		v3.GET("/10", aip.Lab10Code)
	}

	router.Run(":8080")
}
