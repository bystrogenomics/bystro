package decompress

import (
	"archive/tar"
	"bufio"
	"errors"
	"fmt"
	"io"
	"io/fs"
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

	if err == io.ErrUnexpectedEOF {
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

func GetAnnotationFhFromTarArchive(archive *os.File) (BystroReader, fs.FileInfo, error) {
	tarReader, fileStats, err := getTarReader(archive)

	if err != nil {
		return nil, nil, err
	}

	bzfReader, err := getBgzipReaderFromBgzipAnnotation(tarReader)

	if err == nil {
		return bzfReader, fileStats, nil
	}

	archive.Seek(0, 0)

	bufioReader, err := getBufioReaderFromGzipAnnotation(tarReader)

	return bufioReader, fileStats, err
}

func getBgzipReaderFromBgzipAnnotation(tarReader *tar.Reader) (*BzfBystroReader, error) {
	err := pointTarReaderAtAnnotation(tarReader)
	if err != nil {
		return nil, err
	}

	bgzfReader, err := bgzf.NewReader(tarReader, 0)
	if err != nil {
		return nil, err
	}

	return &BzfBystroReader{Reader: bgzfReader}, nil
}

func getBufioReaderFromGzipAnnotation(tarReader *tar.Reader) (*BufioBystroReader, error) {
	err := pointTarReaderAtAnnotation(tarReader)
	if err != nil {
		return nil, err
	}

	gzipReader, err := gzip.NewReader(tarReader)
	if err != nil {
		return nil, err
	}

	bufioReader := bufio.NewReader(gzipReader)

	return &BufioBystroReader{Reader: bufioReader}, nil
}

func getTarReader(archive *os.File) (*tar.Reader, fs.FileInfo, error) {
	fileStats, err := archive.Stat()
	if err != nil {
		return nil, nil, err
	}

	tarReader := tar.NewReader(archive)

	return tarReader, fileStats, nil
}

func pointTarReaderAtAnnotation(tarReader *tar.Reader) error {
	for {
		header, err := tarReader.Next()
		if err != nil {
			if err == io.EOF {
				break // End of archive
			}
			return err
		}

		// TODO @akotlar 2023-11-24: Take the expected file name from the information submitted in the beanstalkd queue message
		if strings.HasSuffix(header.Name, ExpectedAnnotationFileSuffix) {
			return nil
		}
	}

	return fmt.Errorf("couldn't find annotation file in tarball")
}
