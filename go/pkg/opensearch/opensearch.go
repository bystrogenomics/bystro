package opensearch

import (
	"bytes"
	"encoding/json"
	"net/http"
)

func SendToOpenSearch(data []map[string]interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", "http://<opensearch_endpoint>/index_name/_doc", bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	// Add authentication here if needed
	// req.SetBasicAuth(username, password)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Handle response...
	return nil
}
