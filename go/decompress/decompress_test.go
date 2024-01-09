package decompress

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/biogo/hts/bgzf"
	"github.com/klauspost/compress/gzip"
)

// https://stackoverflow.com/questions/22892120/how-to-generate-a-random-string-of-a-fixed-length-in-go
const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func randStringBytes(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return string(b)
}

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
	line, err := readLineBgzip(reader)
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

func TestReadLinesDefaultBuffer(t *testing.T) {
	testReadLinesWithBuffer(100_000, DEFAULT_BUFFER_SIZE)
}

func TestReadLinesSmallBuffers1(t *testing.T) {
	log.Print("Testing ReadLines with buffer of 1 byte")
	testReadLinesWithBuffer(100_000, 1)
	log.Print("Testing ReadLines with buffer of 10 bytes")
	testReadLinesWithBuffer(100_000, 100)
}

func TestReadLinesSmallBuffers1Parallel(t *testing.T) {
	log.Print("Testing ReadLines with buffer of 1 byte")
	go testReadLinesWithBuffer(100_000, 1)
	log.Print("Testing ReadLines with buffer of 10 bytes")
	testReadLinesWithBuffer(100_000, 100)
}

func TestReadLinesSmallBuffers2(t *testing.T) {
	log.Print("Testing ReadLines with buffer of 400 bytes")
	testReadLinesWithBuffer(100_000, 400)
	log.Print("Testing ReadLines with buffer of 1000 bytes")
	testReadLinesWithBuffer(100_000, 1000)
}

func TestReadLinesSmallBuffers2Parallel(t *testing.T) {
	log.Print("Testing ReadLines with buffer of 400 bytes")
	go testReadLinesWithBuffer(100_000, 400)
	log.Print("Testing ReadLines with buffer of 1000 bytes")
	testReadLinesWithBuffer(100_000, 1000)
}

func testReadLinesWithBuffer(numLines int, bufferSize int) {
	// Step 1: Generate Test Data
	testDataBytes := generateTestData(numLines)

	// Step 2: Compress the Data
	random_name := randStringBytes(10)
	compressedFileName := fmt.Sprintf("test_data_%s.bgz", random_name)
	compressData(testDataBytes, compressedFileName)

	// Step 3: Read the Compressed Data
	expectedLines := strings.Split(string(testDataBytes), "\n")
	testDataBytes = nil

	err := readCompressedData(compressedFileName, expectedLines, bufferSize)

	// delete the file
	err2 := os.Remove(compressedFileName)
	if err != nil {
		log.Fatalf("Error reading compressed data: %v", err)
	}

	if err2 != nil {
		log.Fatalf("Error deleting compressed file: %v", err2)
	}
}

func generateTestData(numLines int) []byte {
	min := 1
	max := 3000

	var buffer bytes.Buffer
	for i := 0; i < numLines; i++ {
		// generate a random line

		buffer.WriteString(fmt.Sprintf("Line %d %s\n", i, randStringBytes(rand.Intn(max-min)+min)))
	}
	return buffer.Bytes()
}

func compressData(data []byte, fileName string) {
	f, _ := os.Create(fileName)
	defer f.Close()

	w := bgzf.NewWriter(f, 1)
	defer w.Close()

	w.Write(data)
}

func readCompressedData(fileName string, expectedLines []string, bufferSize int) error {
	f, _ := os.Open(fileName)
	defer f.Close()

	r, _ := bgzf.NewReader(f, 0)
	defer r.Close()

	i := 0
	for {
		lines, err := readLinesBgzipWithBuffer(r, bufferSize)

		if len(lines) > 0 {
			strLines := strings.Split(string(lines), "\n")

			for _, line := range strLines {
				if line != expectedLines[i] {
					return fmt.Errorf("Line %d does not match. Got: %s, expected: %s", i, line, expectedLines[i])
				}

				i += 1
			}
		}

		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}
