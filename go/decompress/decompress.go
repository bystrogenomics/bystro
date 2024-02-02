package decompress

import (
	"bufio"
	"errors"
	"io"
	"log"
	"os"
	"strings"

	"github.com/biogo/hts/bgzf"
	gzip "github.com/klauspost/pgzip"
)

const ExpectedAnnotationFileSuffix = "annotation.tsv.gz"
const DefaultBufferSize = 64 * 1024 * 8 // 8 bgzip blocks at a time

var ErrBufferSize = errors.New("bufferSize must be greater than 0")

type BystroReader interface {
	ReadLines() ([]byte, error)
	ReadLine() ([]byte, error)
}

type BzfBystroReader struct {
	Reader     *bgzf.Reader
	BufferSize uint
}

type BufioBystroReader struct {
	Reader     *bufio.Reader
	BufferSize uint
}

// Read a line up to the next newline character, and return the line excluding the newline character
// Implementation follows the example in the bgzf package documentation: https://github.com/biogo/hts/blob/bb1e21d1bfc7f2b1e124ca0a1ed98493d191db78/bgzf/line_example_test.go#L70
func readLineBgzipNoTx(r *bgzf.Reader) ([]byte, error) {
	var (
		data []byte
		b    byte
		err  error
	)
	for {
		b, err = r.ReadByte()
		if err != nil {
			break
		}
		if b == '\n' {
			break
		}
		data = append(data, b)
	}
	return data, err
}

func readLineBgzip(r *bgzf.Reader) ([]byte, error) {
	tx := r.Begin()
	data, err := readLineBgzipNoTx(r)
	tx.End()
	return data, err
}

func readLinesBgzipWithBuffer(r *bgzf.Reader, bufferSize uint) ([]byte, error) {
	if bufferSize == 0 {
		return nil, ErrBufferSize
	}

	buf := make([]byte, bufferSize)

	tx := r.Begin()
	defer tx.End()

	bytesRead, err := r.Read(buf)

	if bytesRead == 0 {
		return nil, err
	}

	if err != nil {
		if buf[bytesRead-1] != '\n' {
			return buf[:bytesRead], err
		}

		return buf[:bytesRead-1], err
	}

	// Since not at EOF, we know that there is more to read
	if buf[bytesRead-1] != '\n' {
		remainder, err := readLineBgzipNoTx(r)
		return append(buf[:bytesRead], remainder...), err
	}
	// last byte is newline
	return buf[:bytesRead-1], err
}

func readLine(r *bufio.Reader) ([]byte, error) {
	var (
		data []byte
		b    byte
		err  error
	)
	for {
		b, err = r.ReadByte()
		if err != nil {
			break
		}
		if b == '\n' {
			break
		}
		data = append(data, b)
	}
	return data, err
}

func readLinesWithBuffer(r *bufio.Reader, bufferSize uint) ([]byte, error) {
	if bufferSize == 0 {
		return nil, ErrBufferSize
	}

	buf := make([]byte, bufferSize)

	bytesRead, err := io.ReadFull(r, buf)

	if bytesRead == 0 {
		return nil, err
	}

	// The typical errors will io.ErrUnexpectedEOF (buffer longer than input) and io.EOF (file ended)
	if err != nil {
		// Ensure that bgzip and gzip implementations are consistent
		if err == io.ErrUnexpectedEOF {
			err = io.EOF
		}

		if buf[bytesRead-1] == '\n' {
			return buf[:bytesRead-1], err
		}

		return buf[:bytesRead], err
	}

	if buf[bytesRead-1] != '\n' {
		remainder, err := readLine(r)
		return append(buf[:bytesRead], remainder...), err
	}
	// last byte is newline
	return buf[:bytesRead-1], err
}

func (r *BzfBystroReader) ReadLines() ([]byte, error) {
	return readLinesBgzipWithBuffer(r.Reader, DefaultBufferSize)
}

func (r *BzfBystroReader) ReadLine() ([]byte, error) {
	return readLineBgzip(r.Reader)
}

func (r *BufioBystroReader) ReadLines() ([]byte, error) {
	return readLinesWithBuffer(r.Reader, DefaultBufferSize)
}
func (r *BufioBystroReader) ReadLine() ([]byte, error) {
	return r.Reader.ReadBytes('\n')
}

func GetHeaderPaths(b BystroReader) ([][]string, []string) {
	line, err := b.ReadLine()
	if err != nil {
		log.Fatalf("Error reading header line due to: [%s]\n", err)
	}

	headers := strings.Fields(string(line))

	headerPaths := [][]string{}

	for _, header := range headers {
		path := strings.Split(header, ".")

		headerPaths = append(headerPaths, path)
	}

	return headerPaths, headers
}

func GetAnnotationFh(input *os.File) (BystroReader, error) {
	bzfReader, err := getBgzipReaderFromBgzipAnnotation(input)

	if err == nil {
		return bzfReader, nil
	}

	input.Seek(0, 0)

	bufioReader, err := getBufioReaderFromGzipAnnotation(input)

	return bufioReader, err
}

func getBgzipReaderFromBgzipAnnotation(inputFh *os.File) (*BzfBystroReader, error) {
	bgzfReader, err := bgzf.NewReader(inputFh, 0)
	if err != nil {
		return nil, err
	}

	return &BzfBystroReader{Reader: bgzfReader}, nil
}

func getBufioReaderFromGzipAnnotation(inputFh *os.File) (*BufioBystroReader, error) {
	gzipReader, err := gzip.NewReader(inputFh)
	if err != nil {
		return nil, err
	}

	bufioReader := bufio.NewReader(gzipReader)

	return &BufioBystroReader{Reader: bufioReader}, nil
}
