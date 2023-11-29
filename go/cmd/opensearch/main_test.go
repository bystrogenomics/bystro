package main

import (
	"bufio"
	"bytes"
	"strings"
	"testing"

	"github.com/biogo/hts/bgzf"
	"github.com/klauspost/compress/gzip"
)

// Helper function to create a bgzip compressed data from a string
func createBgzipData(data string) (*bytes.Buffer, error) {
	var buffer bytes.Buffer
	writer := bgzf.NewWriter(&buffer, 1)

	_, err := writer.Write([]byte(data))
	if err != nil {
		writer.Close()
		return nil, err
	}

	err = writer.Close()
	if err != nil {
		return nil, err
	}

	return &buffer, nil
}

// Helper function to create a bgzip compressed data from a string
func createGzipData(data string) (*bytes.Buffer, error) {
	var buffer bytes.Buffer
	writer := gzip.NewWriter(&buffer)

	_, err := writer.Write([]byte(data))
	if err != nil {
		writer.Close()
		return nil, err
	}

	err = writer.Close()
	if err != nil {
		return nil, err
	}

	return &buffer, nil
}

func Test_readLineBgzip(t *testing.T) {
	testData := "hello\nworld"
	compressedData, err := createBgzipData(testData)
	if err != nil {
		t.Fatalf("Failed to create compressed bgzip data: %v", err)
	}

	reader, err := bgzf.NewReader(compressedData, 1)
	if err != nil {
		t.Fatalf("Failed to create bgzf.Reader: %v", err)
	}

	// Test the function
	line, err := _readLineBgzip(reader)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if string(line) != "hello" {
		t.Errorf("Expected 'hello', got '%s'", string(line))
	}
}

func Test_readLineGzip(t *testing.T) {
	testData := "hello\nworld"
	compressedData, err := createGzipData(testData)
	if err != nil {
		t.Fatalf("Failed to create compressed bgzip data: %v", err)
	}

	reader, err := gzip.NewReader(compressedData)
	if err != nil {
		t.Fatalf("Failed to create bgzf.Reader: %v", err)
	}

	bufioReader := bufio.NewReader(reader)

	// Test the function
	line, err := readLine(bufioReader)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if string(line) != "hello" {
		t.Errorf("Expected 'hello', got '%s'", string(line))
	}
}

func Test_readLine(t *testing.T) {
	// Create a bufio.Reader with test data
	testData := "test line\nanother line"
	reader := bufio.NewReader(strings.NewReader(testData))

	// Test the function
	line, err := readLine(reader)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if string(line) != "test line" {
		t.Errorf("Expected 'test line', got '%s'", string(line))
	}
}
