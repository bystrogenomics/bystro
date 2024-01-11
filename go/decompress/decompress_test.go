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
	"sync"
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
	testReadLinesWithBuffer(100_000, DefaultBufferSize)
}

func TestReadLinesSmallBuffers1(t *testing.T) {
	log.Print("Testing ReadLines with buffer of 1 byte")
	testReadLinesWithBuffer(100_000, 1)
	log.Print("Testing ReadLines with buffer of 10 bytes")
	testReadLinesWithBuffer(100_000, 100)
}

func TestReadLinesSmallBuffers1Parallel(t *testing.T) {
	// Test if the code is re-entrant
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		testReadLinesWithBuffer(100_000, 1)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		testReadLinesWithBuffer(100_000, 100)
	}()

	wg.Wait()
}

func TestReadLinesSmallBuffers2(t *testing.T) {
	log.Print("Testing ReadLines with buffer of 400 bytes")
	testReadLinesWithBuffer(100_000, 400)
	log.Print("Testing ReadLines with buffer of 1000 bytes")
	testReadLinesWithBuffer(100_000, 1000)
}

func TestReadLinesSmallBuffers2Parallel(t *testing.T) {
	// Test if the code is re-entrant
	log.Print("Testing ReadLines with buffer of 400 and 1000 bytes, run in parallel")
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		testReadLinesWithBuffer(100_000, 400)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		testReadLinesWithBuffer(100_000, 1000)
	}()

	wg.Wait()
}

func testReadLinesWithBuffer(numLines int, bufferSize uint) {
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

func readCompressedData(fileName string, expectedLines []string, bufferSize uint) error {
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

// NOTE: test data comes from `../test/opensearch/testdata/input.txt`
func Test_readLinesWithBuffer(t *testing.T) {
	type args struct {
		bufferSize uint
		input      []byte
	}
	type want struct {
		file   string
		golden []byte
	}
	tests := []struct {
		name    string
		args    args
		want    []byte
		wantErr []error
	}{
		{
			name: "empty input",
			args: args{
				bufferSize: 100,
				input:      []byte(""),
			},
			want:    []byte(""),
			wantErr: []error{io.EOF, io.EOF, io.EOF},
		},
		{
			name: "0 buffer",
			args: args{
				bufferSize: 0,
				input:      []byte(""),
			},
			want:    []byte(""),
			wantErr: []error{ErrBufferSize, ErrBufferSize, ErrBufferSize},
		},
		{
			name: "When buffer is not as long as input we expect no errors",
			args: args{
				bufferSize: 1,
				input:      []byte("a\tb.c\td.e.f\n1\tA\t1;2|3;4/5\n1|2\tA\t1;2|3;4/5\na/2\tA;B\t1/2|3;4/5\n"),
			},
			want:    []byte("a\tb.c\td.e.f"),
			wantErr: []error{nil, nil, nil},
		},
		{
			name: "When buffer is longer than input, we expected either ErrUnexpectedEOF (bufio read) or EOF (bgzip read)",
			args: args{
				bufferSize: 100,
				input:      []byte("a\tb.c\td.e.f\n1\tA\t1;2|3;4/5\n1|2\tA\t1;2|3;4/5\na/2\tA;B\t1/2|3;4/5\n"),
			},
			want:    []byte("a\tb.c\td.e.f\n1\tA\t1;2|3;4/5\n1|2\tA\t1;2|3;4/5\na/2\tA;B\t1/2|3;4/5"),
			wantErr: []error{io.EOF, io.EOF, io.EOF},
		},
		{
			name: "We do not truncate the input when newline is missing from end of file",
			args: args{
				bufferSize: 100,
				input:      []byte("a\tb.c\td.e.f\n1\tA\t1;2|3;4/5\n1|2\tA\t1;2|3;4/5\na/2\tA;B\t1/2|3;4/5"),
			},
			want:    []byte("a\tb.c\td.e.f\n1\tA\t1;2|3;4/5\n1|2\tA\t1;2|3;4/5\na/2\tA;B\t1/2|3;4/5"),
			wantErr: []error{io.EOF, io.EOF, io.EOF},
		},
		{
			name: "Extra newlines within middle of file are retained",
			args: args{
				bufferSize: 100,
				input:      []byte("a\tb.c\td.e.f\n\n\n1\tA\t1;2|3;4/5\n1|2\tA\t1;2|3;4/5\na/2\tA;B\t1/2|3;4/5\n"),
			},
			want:    []byte("a\tb.c\td.e.f\n\n\n1\tA\t1;2|3;4/5\n1|2\tA\t1;2|3;4/5\na/2\tA;B\t1/2|3;4/5"),
			wantErr: []error{io.EOF, io.EOF, io.EOF},
		},
		{
			name: "We support large buffers",
			args: args{
				bufferSize: 1_000_000,
				input:      []byte("a\tb.c\td.e.f\n\n\n1\tA\t1;2|3;4/5\n1|2\tA\t1;2|3;4/5\na/2\tA;B\t1/2|3;4/5\n"),
			},
			want:    []byte("a\tb.c\td.e.f\n\n\n1\tA\t1;2|3;4/5\n1|2\tA\t1;2|3;4/5\na/2\tA;B\t1/2|3;4/5"),
			wantErr: []error{io.EOF, io.EOF, io.EOF},
		},
	}

	for _, tt := range tests {
		log.Println("Running decompressed IO test: ", tt.name)
		b := bytes.NewReader(tt.args.input)
		r := bufio.NewReader(b)
		got, err := readLinesWithBuffer(r, tt.args.bufferSize)

		if err != tt.wantErr[0] {
			log.Fatalf("Expected error: %v\ngot: %v", tt.wantErr, err)
		}

		if !bytes.Equal(got, tt.want) {
			log.Fatalf("Expected:\n%v\ngot:\n%v", string(tt.want), string(got))
		}
	}

	for _, tt := range tests {
		log.Println("Running gzip test: ", tt.name)

		// Compress the data
		var buf bytes.Buffer
		writer := gzip.NewWriter(&buf)

		if _, err := writer.Write(tt.args.input); err != nil {
			log.Fatal(err)
		}
		if err := writer.Close(); err != nil {
			log.Fatal(err)
		}

		b := bytes.NewReader(buf.Bytes())
		r, err := gzip.NewReader(b)
		if err != nil {
			log.Fatal(err)
		}
		bufioReader := bufio.NewReader(r)

		got, err := readLinesWithBuffer(bufioReader, tt.args.bufferSize)

		if err != tt.wantErr[1] {
			log.Fatalf("Expected error: %v\ngot: %v", tt.wantErr, err)
		}

		if !bytes.Equal(got, tt.want) {
			log.Fatalf("Expected:\n%v\ngot:\n%v", string(tt.want), string(got))
		}
	}

	for _, tt := range tests {
		log.Println("Running bgzip test: ", tt.name)

		var buf bytes.Buffer
		writer := bgzf.NewWriter(&buf, 1)

		if _, err := writer.Write(tt.args.input); err != nil {
			log.Fatal(err)
		}
		if err := writer.Close(); err != nil {
			log.Fatal(err)
		}

		b := bytes.NewReader(buf.Bytes())
		r, err := bgzf.NewReader(b, 0)
		if err != nil {
			log.Fatal(err)
		}

		got, err := readLinesBgzipWithBuffer(r, tt.args.bufferSize)

		if err != tt.wantErr[2] {
			log.Fatalf("Expected error: %v, got: %v", tt.wantErr[1], err)
		}

		if !bytes.Equal(got, tt.want) {
			log.Fatalf("Expected: %v, got: %v", tt.want, got)
		}
	}
}
