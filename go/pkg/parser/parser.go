package parser

import (
	"bytes"
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/opensearch-project/opensearch-go"
)

type Job struct {
	Lines []string
	Start int
}

const (
	LINE_DELIMITER      = "\n"
	FIELD_DELIMITER     = "\t"
	OVERLAP_DELIMITER   = "/"
	POSITION_DELIMITER  = "|"
	VALUE_DELIMITER     = ";"
	MISSING_FIELD_VALUE = "NA"
)

// Parse takes the file content and converts it into an array of nested maps.
func Parse(headerPaths [][]string, indexName string, osConfig opensearch.Config, workQueue chan Job, done chan bool, channelId int) {
	client, err := opensearch.NewClient(osConfig)

	if err != nil {
		log.Fatal(err)
	}

	var w = bytes.NewBuffer(nil)
	var enc = sonic.ConfigDefault.NewEncoder(w)

	outerMap := make(map[string]map[string]string)
	innerMap1 := make(map[string]string)
	outerMap["index"] = innerMap1

	for job := range workQueue {
		lines := job.Lines
		id := job.Start - 1
		for _, line := range lines {
			id += 1

			fields := strings.Split(line, FIELD_DELIMITER)
			if len(fields) != len(headerPaths) {
				log.Fatal("fields and headerPaths are not the same length")
			}

			nestedMap := buildNestedMap(headerPaths, fields)

			innerMap1["_index"] = indexName
			innerMap1["_id"] = strconv.Itoa(id)

			enc.Encode(outerMap)
			enc.Encode(nestedMap)
		}

		res, err := client.Bulk(strings.NewReader(w.String()))

		if err != nil || res.IsError() {
			log.Fatal(err, res.StatusCode)
		}

		if err != nil || res.IsError() {
			log.Fatal(err, res.StatusCode)
		}

		res.Body.Close()
		w.Reset()

		fmt.Println("Channel id:", channelId, "Processed from ", job.Start, " to ", id, " with ", len(lines), " lines")
	}

	done <- true
}

func buildFlatMap(headerFields []string, values []string) map[string]any {
	row := make(map[string]any)
	// fmt.Println("len(headerPaths)", len(headerPaths))
	for i, fieldName := range headerFields {
		row[fieldName] = ensure3DArray(values[i])
	}

	return row
}

// buildNestedMap constructs a nested map based on the headers and values.
func buildNestedMap(headerPaths [][]string, values []string) map[string]any {
	nestedMap := make(map[string]any)
	for i, headerPath := range headerPaths {
		currentMap := nestedMap
		for j, node := range headerPath {
			if j == len(headerPath)-1 {
				currentMap[node] = ensure3DArray(values[i])
			} else {
				if _, exists := currentMap[node]; !exists {
					currentMap[node] = make(map[string]any)
				}
				currentMap = currentMap[node].(map[string]any)
			}
		}
	}

	return nestedMap
}

// ensure3DArray ensures that the value is converted to a 3D array.
func ensure3DArray(value string) any {
	var outerArray [][][]any

	for _, positionValues := range strings.Split(value, POSITION_DELIMITER) {
		var middleArray [][]any

		for _, valueValues := range strings.Split(positionValues, VALUE_DELIMITER) {
			var innerArray []any

			if !strings.Contains(valueValues, OVERLAP_DELIMITER) {
				item := processItem(valueValues)
				slice := []any{item}

				middleArray = append(middleArray, slice)
				continue
			}

			for _, thirdPart := range strings.Split(valueValues, OVERLAP_DELIMITER) {
				innerArray = append(innerArray, processItem(thirdPart))
			}

			middleArray = append(middleArray, innerArray)
		}

		outerArray = append(outerArray, middleArray)
	}

	return outerArray
}

// processItem interprets the given item as per the specified rules.
func processItem(item string) any {
	trimmedItem := strings.TrimSpace(item)

	// Check for "NA" and return nil
	if trimmedItem == MISSING_FIELD_VALUE {
		return nil
	}

	// Try to parse as an integer
	if intVal, err := strconv.Atoi(trimmedItem); err == nil {
		return intVal
	}

	// Try to parse as a float
	if floatVal, err := strconv.ParseFloat(trimmedItem, 64); err == nil {
		return floatVal
	}

	// Default to treating as a string
	return trimmedItem
}
