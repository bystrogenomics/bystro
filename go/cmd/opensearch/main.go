package main

import (
	"archive/tar"
	"bystro/pkg/parser"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/biogo/hts/bgzf"
	"github.com/opensearch-project/opensearch-go"
)

// readNextChunk reads bytes from a BGZF file until it reaches a newline.
// It returns a chunk aligned on a newline boundary.
func readNextChunk(file *bgzf.Reader) []byte {
	const maxChunkSize = 1024 // Define a suitable chunk size

	buf := make([]byte, 0, maxChunkSize)
	temp := make([]byte, 1)

	for len(buf) < maxChunkSize {
		_, err := file.Read(temp)
		if err != nil {
			if err == io.EOF {
				break // End of file reached
			}
			log.Fatal(err) // Handle other errors
		}

		buf = append(buf, temp[0])
		if temp[0] == '\n' {
			break // Newline boundary reached
		}
	}

	return buf
}

func processChunk(chunk []byte, headerPaths [][]string) {
	// Process the chunk here
	parser.Parse(chunk, headerPaths)
}

func main() {
	client, err := opensearch.NewClient(opensearch.Config{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
		Addresses:     []string{"http://10.98.135.70:9200"},
		MaxRetries:    5,
		RetryOnStatus: []int{502, 503, 504},
	})

	client.IndicesCreateRequest{
		Index: "go-test-index1",
		// Body:  settings,
	}
	// Open the tar archive
	archive, err := os.Open("/seqant/user-data/63ddc9ce1e740e0020c39928/6556f106f71022dc49c8e560/output/all_chr1_phase3_shapeit2_mvncall_integrated_v5b_20130502_genotypes_vcf.tar")
	if err != nil {
		panic(err)
	}
	defer archive.Close()

	tarReader := tar.NewReader(archive)

	var b *bgzf.Reader
	for {
		header, err := tarReader.Next()
		if err != nil {
			if err == io.EOF {
				break // End of archive
			}
			panic(err) // Handle other errors
		}

		if strings.HasSuffix(header.Name, "annotation.tsv.gz") {
			// Process this file
			// processCompressedFile(tarReader)
			b, err = bgzf.NewReader(tarReader, 0)
			if err != nil {
				log.Fatal(err)
			}
			break
		}
	}

	// file, err := os.Open("/home/ubuntu/bystro/rust/index_annotation/all_chr1_phase3_shapeit2_mvncall_integrated_v5b_20130502_genotypes_vcf.annotation.tsv.gz")
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// defer file.Close()

	//
	// b, err := bgzf.NewReader(file, 0)
	buf := make([]byte, 0, 1024*1024)
	bytesRead := 0
	for {
		extraByte, err := b.ReadByte()
		if err != nil {
			fmt.Errorf("error reading byte: %v", err)
			break
		}
		buf = append(buf, extraByte)
		bytesRead++
		if extraByte == '\n' {
			fmt.Println("found newline", extraByte)
			break
		}
	}
	headers := strings.Fields(string(buf[:bytesRead]))
	// fmt.Println("headers:", string(buf[:bytesRead]))
	headerPaths := [][]string{}

	for _, header := range headers {
		path := strings.Split(header, ".")

		headerPaths = append(headerPaths, path)
	}
	// fmt.Println("headerPaths:", headerPaths)
	// os.Exit(1)

	if err != nil {
		log.Fatal(err)
	}
	// bytesRead := 0
	// Example usage
	buf = make([]byte, 16*1024)
	// fmt.Println("Size of buffer:", len(buf))
	for {
		// chunk := readNextChunk(file)
		// if len(chunk) == 0 {
		// 	break // No more data to process
		// }
		buf = make([]byte, 16*1024)
		bytesRead, err := b.Read(buf)
		if err != nil {
			if err == io.EOF {
				break
			}
			panic(err)
		}
		// fmt.Println((buf))
		bufRead := buf[:bytesRead]
		if bufRead[len(bufRead)-1] != '\n' {
			if err != nil {
				if err == io.EOF {
					break
				}
				panic(err)
			}
			// fmt.Println("bytesRead:", bytesRead)
			// If the last byte isn't a newline, read until newline is found
			// for buf[bytesRead-1] != '\n' {
			for {
				extraByte, err := b.ReadByte()
				if err != nil {
					break
				}
				bufRead = append(bufRead, extraByte)
				bytesRead++
				if extraByte == '\n' {
					break
				}
			}
		}

		// fmt.Println("bufRead:", len(bufRead), bufRead[len(bufRead)-1])

		// Process the chunk
		go parser.Parse(bufRead, headerPaths)
	}
	// content, err := ioutil.ReadFile("../../test/opensearch/testdata/input.txt")
	// if err != nil {
	// 	panic(err)
	// }

	// nestedMap := parser.Parse(content)

	// // Convert and print the map in a formatted JSON structure
	// formattedJSON, err := json.MarshalIndent(nestedMap, "", "  ")
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println(string(formattedJSON))

	// err = opensearch.SendToOpenSearch(nestedMap)
	// if err != nil {
	// 	fmt.Println("Error sending to OpenSearch:", err)
	// }
}
