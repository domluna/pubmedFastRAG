package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

type RAGRequest struct {
	Query string `json:"query"`
	K     int    `json:"k"`
}

type RAGResponse struct {
	PMID            int    `json:"pmid"`
	Distance        int    `json:"distance"`
	Authors         string `json:"authors"`
	Title           string `json:"title"`
	Abstract        string `json:"abstract"`
	PublicationYear int    `json:"publication_year"`
}

func handleHome(c *gin.Context) {
	c.HTML(http.StatusOK, "index.html", nil)
}

// body is RAGRequest
func handleSearch(c *gin.Context) {
	var req RAGRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Println(err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}
	if req.K > 100 {
		log.Printf("Request for k = %d truncated to 100", req.K)
		req.K = 100
	}

	body, err := json.Marshal(req)
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to marshal request"})
		return
	}

	apiResp, err := http.Post("http://0.0.0.0:8003/find_matches", "application/json", bytes.NewBuffer(body))
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer apiResp.Body.Close()

	respBody, err := io.ReadAll(apiResp.Body)
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read response"})
		return
	}

	var data []RAGResponse
	if err := json.Unmarshal(respBody, &data); err != nil {
		log.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse response"})
		return
	}

	c.JSON(http.StatusOK, data)
}

func main() {
	r := gin.Default()
	r.LoadHTMLGlob("templates/*")
	r.GET("/", handleHome)
	r.POST("/search", handleSearch)
	log.Fatal(r.Run(":8080"))
}
