package main

import (
	"bytes"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math/bits"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

type RAGServer struct {
	IDs  []int64
	Data [][]uint64
	DB   *sql.DB
}

func NewRAGServer(dbpath string) (*RAGServer, error) {
	db, err := sql.Open("sqlite3", dbpath)
	if err != nil {
		return nil, err
	}

	ids, data, err := loadDataFromBin()
	if err != nil {
		return nil, err
	}

	return &RAGServer{
		IDs:  ids,
		Data: data,
		DB:   db,
	}, nil
}

func loadDataFromBin() ([]int64, [][]uint64, error) {
	dataFile, err := os.ReadFile("../bindata/data.bin")
	if err != nil {
		return nil, nil, err
	}

	numRows := binary.LittleEndian.Uint64(dataFile[:8])
	numColumns := binary.LittleEndian.Uint64(dataFile[8:16])

	fmt.Println(numRows, numColumns)

	data := make([][]uint64, numRows)
	for i := range data {
		data[i] = make([]uint64, numColumns)
		for j := range data[i] {
			data[i][j] = binary.LittleEndian.Uint64(dataFile[16+(i*int(numColumns)+j)*8 : 16+(i*int(numColumns)+j+1)*8])
		}
	}

	idsFile, err := os.ReadFile("../bindata/ids.bin")
	if err != nil {
		return nil, nil, err
	}

	ids := make([]int64, numRows)
	for i := range ids {
		ids[i] = int64(binary.LittleEndian.Uint64(idsFile[i*8 : (i+1)*8]))
	}

	return ids, data, nil
}

func getArticleData(db *sql.DB, pmids []string) (map[string]map[string]interface{}, error) {
	placeholders := make([]string, len(pmids))
	args := make([]interface{}, len(pmids))
	for i, pmid := range pmids {
		placeholders[i] = "?"
		args[i] = pmid
	}

	log.Println("Querying with PMIDs:", pmids)
	query := fmt.Sprintf("SELECT pmid, title, authors, abstract, publication_year FROM articles WHERE pmid IN (%s)", strings.Join(placeholders, ","))

	rows, err := db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("error executing query: %v", err)
	}
	defer rows.Close()

	articleData := make(map[string]map[string]interface{})
	for rows.Next() {
		var pmid sql.NullString
		var title sql.NullString
		var authors sql.NullString
		var abstract sql.NullString
		var publicationYear sql.NullInt64

		err := rows.Scan(&pmid, &title, &authors, &abstract, &publicationYear)
		if err != nil {
			return nil, fmt.Errorf("error scanning row: %v", err)
		}

		// Use empty string for NULL values
		pmidStr := ""
		if pmid.Valid {
			pmidStr = pmid.String
		}

		articleData[pmidStr] = map[string]interface{}{
			"pmid":             pmidStr,
			"title":            getStringValue(title),
			"authors":          getStringValue(authors),
			"abstract":         getStringValue(abstract),
			"publication_year": getInt64Value(publicationYear),
		}
	}

	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error after scanning all rows: %v", err)
	}

	return articleData, nil
}

// Helper function to handle sql.NullString
func getStringValue(s sql.NullString) string {
	if s.Valid {
		return s.String
	}
	return ""
}

// Helper function to handle sql.NullInt64
func getInt64Value(i sql.NullInt64) int64 {
	if i.Valid {
		return i.Int64
	}
	return 0
}

type EmbedResponse struct {
	Embedding       []float32 `json:"embedding"`
	BinaryEmbedding []uint8   `json:"binary_embedding"`
}

type FindMatchesRequest struct {
	Query string `json:"query"`
	K     int    `json:"k"`
}

type FindMatchesResponse struct {
	PMID            int    `json:"pmid"`
	Distance        int    `json:"distance"`
	Authors         string `json:"authors"`
	Title           string `json:"title"`
	Abstract        string `json:"abstract"`
	PublicationYear int    `json:"publication_year"`
}

func startServer(rag *RAGServer, port int) {
	http.HandleFunc("/find_matches", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req FindMatchesRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON body", http.StatusBadRequest)
			return
		}

		if req.K <= 0 || req.K > 100 {
			http.Error(w, "'k' parameter must be a positive integer and <= 100", http.StatusBadRequest)
			return
		}

		// Call embed service
		embedResp, err := http.Post("http://0.0.0.0:8002/embed", "application/json", bytes.NewBufferString(fmt.Sprintf(`{"text":"%s"}`, req.Query)))
		if err != nil {
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}
		defer embedResp.Body.Close()

		if embedResp.StatusCode != http.StatusOK {
			http.Error(w, fmt.Sprintf("Error from embed service: %s", embedResp.Status), http.StatusBadGateway)
			return
		}

		var embedBody EmbedResponse
		if err := json.NewDecoder(embedResp.Body).Decode(&embedBody); err != nil {
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}

		if len(embedBody.BinaryEmbedding) != 64 {
			http.Error(w, "Embedded query must be an array of 64 UInt8 elements", http.StatusBadGateway)
			return
		}

		queryUint64 := make([]uint64, 8)
		for i := 0; i < 8; i++ {
			queryUint64[i] = binary.LittleEndian.Uint64(embedBody.BinaryEmbedding[i*8 : (i+1)*8])
		}

		kClosestStart := time.Now()
		results := kClosestParallel(rag.Data, queryUint64, req.K)
		log.Printf("kClosestParallel took %v querying %d results\n", time.Since(kClosestStart), req.K)

		// Fetch article data from SQLite
		pmids := make([]string, len(results))
		for i, result := range results {
			pmids[i] = fmt.Sprintf("%d", rag.IDs[result.second])
		}

		articleData, err := getArticleData(rag.DB, pmids)
		if err != nil {
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}

		response := make([]FindMatchesResponse, 0, len(results))
		for _, r := range results {
			id := fmt.Sprintf("%d", rag.IDs[r.second])
			if a, ok := articleData[id]; ok {
				response = append(response, FindMatchesResponse{
					PMID:            int(rag.IDs[r.second]),
					Distance:        r.first,
					Authors:         a["authors"].(string),
					Title:           a["title"].(string),
					Abstract:        a["abstract"].(string),
					PublicationYear: int(a["publication_year"].(int64)),
				})
			}
		}

		json.NewEncoder(w).Encode(response)
	})

	fmt.Printf("Server starting on port %d\n", port)
	http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
}

func hammingDistance(x1, x2 []uint64) int {
	s := 0
	for i := range x1 {
		s += bits.OnesCount64(x1[i] ^ x2[i])
	}
	return s
}

type MaxHeap struct {
	data       []Pair
	currentIdx int
	k          int
}

type Pair struct {
	first  int
	second int
}

func NewMaxHeap(k int) *MaxHeap {
	data := make([]Pair, k)
	for i := range data {
		data[i] = Pair{first: int(^uint(0) >> 1), second: -1}
	}
	return &MaxHeap{
		data:       data,
		currentIdx: 0,
		k:          k,
	}
}

func (h *MaxHeap) Insert(value Pair) {
	if h.currentIdx < h.k {
		h.data[h.currentIdx] = value
		h.currentIdx++
		if h.currentIdx == h.k {
			h.makeHeap()
		}
	} else if value.first < h.data[0].first {
		h.data[0] = value
		h.heapify(0)
	}
}

func (h *MaxHeap) makeHeap() {
	for i := h.k/2 - 1; i >= 0; i-- {
		h.heapify(i)
	}
}

func (h *MaxHeap) heapify(i int) {
	largest := i
	left := 2*i + 1
	right := 2*i + 2

	if left < h.k && h.data[left].first > h.data[largest].first {
		largest = left
	}

	if right < h.k && h.data[right].first > h.data[largest].first {
		largest = right
	}

	if largest != i {
		h.data[i], h.data[largest] = h.data[largest], h.data[i]
		h.heapify(largest)
	}
}

func kClosest(db [][]uint64, query []uint64, k int, startind int) []Pair {
	heap := NewMaxHeap(k)
	for i := range db {
		d := hammingDistance(db[i], query)
		heap.Insert(Pair{first: d, second: startind + i})
	}
	sort.Slice(heap.data, func(i, j int) bool {
		return heap.data[i].first < heap.data[j].first
	})
	return heap.data
}

func kClosestParallel(db [][]uint64, query []uint64, k int) []Pair {
	n := len(db)
	if n < 10000 || runtime.GOMAXPROCS(0) == 1 {
		return kClosest(db, query, k, 0)
	}

	numWorkers := runtime.GOMAXPROCS(0)
	chunkSize := n / numWorkers
	var wg sync.WaitGroup
	results := make([][]Pair, numWorkers)

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			start := i * chunkSize
			end := start + chunkSize
			if i == numWorkers-1 {
				end = n
			}
			results[i] = kClosest(db[start:end], query, k, start)
		}(i)
	}

	wg.Wait()

	allResults := make([]Pair, 0, k*numWorkers)
	for _, result := range results {
		allResults = append(allResults, result...)
	}

	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].first < allResults[j].first
	})

	if len(allResults) > k {
		allResults = allResults[:k]
	}

	return allResults
}

func main() {
	rag, err := NewRAGServer("../databases/pubmed_data.db")
	if err != nil {
		panic(err)
	}
	startServer(rag, 8003)
}
