package parser

import (
	"fmt"
	"strconv"
	"strings"
)

// Parse takes the file content and converts it into an array of nested maps.
func Parse(content []byte, headerFields []string) []map[string]interface{} {
	lines := strings.Split(string(content), "\n")
	var result []map[string]interface{}

	var row map[string]interface{}
	for _, line := range lines {
		if line == "" {
			continue
		}
		fields := strings.Split(line, "\t")
		if len(fields) != len(headerFields) {
			fmt.Println("fields:", fields)
			fmt.Println("headerPaths:", headerFields)
			panic("fields and headerPaths are not the same length")
		}
		row = buildFlatMap(headerFields, fields)
		// nestedMap := buildNestedMap(headerPaths, fields)
		result = append(result, row)
	}

	// fmt.Println("result:", result)

	// clear(result)

	return result
}

func buildFlatMap(headerFields []string, values []string) map[string]interface{} {
	row := make(map[string]interface{})
	// fmt.Println("len(headerPaths)", len(headerPaths))
	for i, fieldName := range headerFields {
		row[fieldName] = ensure3DArray(values[i])
	}

	return row
}

// buildNestedMap constructs a nested map based on the headers and values.
func buildNestedMap(headerPaths [][]string, values []string) map[string]interface{} {
	nestedMap := make(map[string]interface{})
	// fmt.Println("len(headerPaths)", len(headerPaths))
	for i, headerPath := range headerPaths {
		currentMap := nestedMap
		// fmt.Println("headerPath", headerPath)
		// fmt.Println("headerPath", headerPath, len(headerPath))
		for j, node := range headerPath {
			// fmt.Println("node", node, len(node))
			if j == len(headerPath)-1 {
				currentMap[node] = ensure3DArray(values[i])
			} else {
				if _, exists := currentMap[node]; !exists {
					currentMap[node] = make(map[string]interface{})
				}
				currentMap = currentMap[node].(map[string]interface{})
			}
		}
	}

	return nestedMap
}

// ensure3DArray ensures that the value is converted to a 3D array.
func ensure3DArray(value string) interface{} {
	var outerArray [][][]interface{}

	for _, positionValues := range strings.Split(value, "|") {
		var middleArray [][]interface{}

		for _, valueValues := range strings.Split(positionValues, ";") {
			var innerArray []interface{}

			if !strings.Contains(valueValues, "/") {
				item := processItem(valueValues)
				slice := []interface{}{item}

				middleArray = append(middleArray, slice)
				continue
			}

			for _, thirdPart := range strings.Split(valueValues, "/") {
				innerArray = append(innerArray, processItem(thirdPart))
			}

			middleArray = append(middleArray, innerArray)
		}

		outerArray = append(outerArray, middleArray)
	}

	return outerArray
}

// processItem interprets the given item as per the specified rules.
func processItem(item string) interface{} {
	trimmedItem := strings.TrimSpace(item)

	// Check for "NA" and return nil
	if trimmedItem == "NA" {
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
