package main

import (
	// "archive/tar"
	"archive/tar"
	"bystro/pkg/parser"
	"context"
	"crypto/tls"
	"flag"
	"math"
	"net/http"
	"runtime"

	// "crypto/tls"
	"fmt"
	"io"
	"log"

	// "net/http"
	"os"
	"strings"

	"github.com/biogo/hts/bgzf"
	"github.com/bytedance/sonic"
	"github.com/opensearch-project/opensearch-go"
	"github.com/opensearch-project/opensearch-go/opensearchapi"
	"gopkg.in/yaml.v3"
	// "github.com/opensearch-project/opensearch-go"
)

type CLIArgs struct {
	annotationTarballPath  string
	osIndexConfigPath      string
	osConnectionConfigPath string
	indexName              string
	allowHTTP              bool
}

type OpensearchNode struct {
	Host   string `yaml:"host"`
	Port   int    `yaml:"port"`
	Scheme string `yaml:"scheme"`
}

type OpensearchConnectionConfig struct {
	Connection struct {
		RequestTimeout int              `yaml:"request_timeout"`
		Gzip           bool             `yaml:"gzip"`
		Nodes          []OpensearchNode `yaml:"nodes"`
	} `yaml:"connection"`
	Auth struct {
		Dummy    bool   `yaml:"dummy"`
		Username string `yaml:"username"`
		Password string `yaml:"password"`
	}
}

type OpensearchMappingConfig struct {
	NumericalFields   []string                  `yaml:"numericalFields" json:"numericalFields"`
	Sort              map[string]string         `yaml:"sort" json:"sort"`
	BooleanFields     []string                  `yaml:"booleanFields" json:"booleanFields"`
	PostIndexSettings map[string]map[string]any `yaml:"post_index_settings" json:"post_index_settings"`
	Settings          map[string]map[string]any `yaml:"index_settings" json:"index_settings"`
	Mappings          map[string]map[string]any `yaml:"mappings" json:"mappings"`
}

func createAddresses(config OpensearchConnectionConfig) []string {
	var addresses []string

	for _, node := range config.Connection.Nodes {
		address := fmt.Sprintf("%s://%s:%d", node.Scheme, node.Host, node.Port)
		addresses = append(addresses, address)
	}

	return addresses
}

func setup(args []string) *CLIArgs {
	cliargs := &CLIArgs{}
	flag.StringVar(&cliargs.annotationTarballPath, "in", "example.tar", "The path to the input tarball")
	flag.StringVar(&cliargs.osIndexConfigPath, "index", "/home/ubuntu/bystro/config/hg19.mapping.yml", "The path to the OpenSearch mapping and index definition (e.g. hg19.mapping.yml)")
	flag.StringVar(&cliargs.osConnectionConfigPath, "connection", "/home/ubuntu/bystro/config/elastic-config2.yml", "The path to the OpenSearch connection config (e.g. config/elasticsearch.yml)")
	flag.StringVar(&cliargs.indexName, "name", "test2", "The index name")
	flag.BoolVar(&cliargs.allowHTTP, "http", false, "Allow http connections (else forces https)")

	a := os.Args[1:]
	if args != nil {
		a = args
	}
	flag.CommandLine.Parse(a)

	return cliargs
}

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

func getHeaderPaths(b *bgzf.Reader) [][]string {
	buf := make([]byte, 0, 1024*1024)
	bytesRead := 0
	for {
		extraByte, err := b.ReadByte()
		if err != nil {
			log.Fatal(err)
		}

		buf = append(buf, extraByte)
		bytesRead++
		if extraByte == '\n' {
			fmt.Println("found newline", extraByte)
			break
		}
	}
	headers := strings.Fields(string(buf[:bytesRead]))

	headerPaths := [][]string{}

	for _, header := range headers {
		path := strings.Split(header, ".")

		headerPaths = append(headerPaths, path)
	}

	return headerPaths
}

func createIndex(opensearchConnectionConfigPath string, opensearchIndexConfigPath string, indexName string, fileSize int64) opensearch.Config {
	var osearchConnConfig OpensearchConnectionConfig
	var osearchMapConfig OpensearchMappingConfig

	connectionSettings, err := os.ReadFile(opensearchConnectionConfigPath)
	if err != nil {
		log.Fatalf("Coudln't read: %s due to: %s", opensearchConnectionConfigPath, err)
	}

	err = yaml.Unmarshal(connectionSettings, &osearchConnConfig)
	if err != nil {
		log.Fatalf("Failed to unmarshal search configuration: %v", err)
	}

	indexConfig, err := os.ReadFile(opensearchIndexConfigPath)
	if err != nil {
		log.Fatalf("Coudln't read: %s due to: %s", opensearchIndexConfigPath, err)
	}
	// fmt.Println(string(index_config))
	err = yaml.Unmarshal(indexConfig, &osearchMapConfig)
	if err != nil {
		log.Fatalf("Unmarshal failed: %v", err)
	}

	_, ok := osearchMapConfig.Settings["index"]["number_of_shards"]

	if !ok {
		osearchMapConfig.Settings["index"]["number_of_shards"] = math.Ceil(float64(fileSize) / float64(1e10))
	}

	fmt.Println("number of shards", osearchMapConfig.Settings["index"]["number_of_shards"])

	settings := osearchMapConfig.Settings
	indexSettings, err := sonic.Marshal(settings)
	if err != nil {
		log.Fatalf("Marshaling failed of index settings: %v", err)
	}

	indexMapping, err := sonic.Marshal(osearchMapConfig.Mappings)
	if err != nil {
		log.Fatalf("Marshaling failed of mappings: %v", err)
	}

	requestBody := fmt.Sprintf(`{
			"settings": %s,
			"mappings": %s
		}`, string(indexSettings), string(indexMapping))

	osConfig := opensearch.Config{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
		Addresses:     createAddresses(osearchConnConfig),
		MaxRetries:    5,
		RetryOnStatus: []int{502, 503, 504},
	}

	client, err := opensearch.NewClient(osConfig)

	resp, err := opensearchapi.IndicesExistsRequest{
		Index: []string{indexName},
	}.Do(context.Background(), client)

	if err != nil {
		log.Fatalf("Error deleting index: %s", err)
	}

	if resp.StatusCode == 200 {
		deleteIndex := opensearchapi.IndicesDeleteRequest{
			Index: []string{indexName},
		}

		deleteIndexResponse, err := deleteIndex.Do(context.Background(), client)

		if err != nil || deleteIndexResponse.IsError() {
			log.Fatalf("Error deleting index: %s", err)
		}
	}

	// ctx := context.Background()
	createIndex := opensearchapi.IndicesCreateRequest{
		Index: indexName,
		Body:  strings.NewReader(requestBody),
	}

	createResp, err := createIndex.Do(context.Background(), client)

	// createResp, err := client.Indices.Create(createIndex)
	if err != nil || createResp.IsError() {
		log.Fatalf("Error creating index: %s", err)
	}

	return osConfig
}

func main() {
	cliargs := setup(nil)

	indexName := cliargs.indexName

	// From testing, performance seems maximized at 32 threads for a 4 vCPU AWS instance
	concurrency := runtime.NumCPU() * 8

	// Open the tar archive
	archive, err := os.Open(cliargs.annotationTarballPath) //os.Open("/seqant/user-data/63ddc9ce1e740e0020c39928/6556f106f71022dc49c8e560/output/all_chr1_phase3_shapeit2_mvncall_integrated_v5b_20130502_genotypes_vcf.tar")
	if err != nil {
		log.Fatal(err)
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
			log.Fatal(err) // Handle other errors
		}

		if strings.HasSuffix(header.Name, "annotation.tsv.gz") {
			b, err = bgzf.NewReader(tarReader, 0)
			if err != nil {
				log.Fatal(err)
			}
			break
		}
	}

	// TODO @akotlar 2023-11-24: Get the file size of just the file being indexed; though the tar archive is about the same size as the file, it's not exactly the same
	fileStats, err := archive.Stat()
	if err != nil {
		log.Fatal(err)
	}

	workQueue := make(chan parser.Job)
	complete := make(chan bool)

	headerPaths := getHeaderPaths(b)

	osConfig := createIndex(cliargs.osConnectionConfigPath, cliargs.osIndexConfigPath, cliargs.indexName, fileStats.Size())

	// Spawn threads
	for i := 0; i < concurrency; i++ {
		go parser.Parse(headerPaths, indexName, osConfig, workQueue, complete)
	}

	chunkStart := 0
	var lines []string
	var readBuffer []byte
	var bufRead []byte
	for {
		readBuffer = make([]byte, 1024*1024)
		bytesRead, err := b.Read(readBuffer)
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Fatal(err)
		}

		// fmt.Println((buf))
		bufRead = readBuffer[:bytesRead]
		if bufRead[len(bufRead)-1] != '\n' {
			if err != nil {
				if err == io.EOF {
					break
				}
				log.Fatal(err)
			}

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

		lines = append(lines, strings.Split(string(bufRead), "\n")...)

		if len(lines) >= 100 {
			workQueue <- parser.Job{Lines: lines, Start: chunkStart}

			chunkStart += len(lines)
			lines = nil
		}
	}

	if len(lines) > 0 {
		workQueue <- parser.Job{Lines: lines, Start: chunkStart}
		readBuffer = nil
		lines = nil
	}

	// Indicate to all processing threads that no more work remains
	close(workQueue)

	// Wait for everyone to finish.
	for i := 0; i < concurrency; i++ {
		<-complete
	}

	fmt.Println("Processed this many lines: ", chunkStart)

	// Update the index settings with post_index_settings and refresh the index

}
