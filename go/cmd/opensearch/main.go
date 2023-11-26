package main

import (
	"bystro/pkg/parser"
	"context"
	"crypto/tls"
	"flag"
	"math"
	"net/http"
	"runtime"

	"fmt"
	"io"
	"log"

	"os"
	"strings"

	"github.com/biogo/hts/bgzf"
	"github.com/bytedance/sonic"
	opensearch "github.com/opensearch-project/opensearch-go/v2"
	opensearchapi "github.com/opensearch-project/opensearch-go/v2/opensearchapi"
	"gopkg.in/yaml.v3"
)

type CLIArgs struct {
	annotationTarballPath  string
	osIndexConfigPath      string
	beanstalkConfigPath    string
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

type ProgressData struct {
	progress int
	skipped  int
}

type ProgressMessage struct {
	submissionID string
	data         ProgressData
	event        string `default:"progress"`
}

type ProgressPublisher struct {
	host    string
	port    int
	queue   string
	message ProgressMessage
}

// Expected beanstalkd format
//
//	addresses:
//	  - <host1>:<port1>
//	tubes:
//	  index:
//	    submission: index
//	    events: index_events
//	  ...
type BeanstalkdConfig struct {
	Addresses []string `yaml:"addresses"`
	Tubes     struct {
		Index struct {
			Submission string `yaml:"submission"`
			Events     string `yaml:"events"`
		} `yaml:"index"`
	} `yaml:"tubes"`
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
	flag.StringVar(&cliargs.annotationTarballPath, "in", "/seqant/user-data/63ddc9ce1e740e0020c39928/6556f106f71022dc49c8e560/output/all_chr1_phase3_shapeit2_mvncall_integrated_v5b_20130502_genotypes_vcf.tar", "The path to the input tarball")
	flag.StringVar(&cliargs.osIndexConfigPath, "index", "/home/ubuntu/bystro/config/hg19.mapping.yml", "The path to the OpenSearch mapping and index definition (e.g. hg19.mapping.yml)")
	flag.StringVar(&cliargs.osConnectionConfigPath, "search", "/home/ubuntu/bystro/config/elastic-config2.yml", "The path to the OpenSearch connection config (e.g. config/elasticsearch.yml)")
	flag.StringVar(&cliargs.beanstalkConfigPath, "queue", "/home/ubuntu/bystro/config/beanstalk.yml", "The path to the Beanstalkd queue connection config (e.g. config/beanstalk.yml)")
	flag.StringVar(&cliargs.indexName, "name", "test2", "The index name")
	flag.BoolVar(&cliargs.allowHTTP, "http", false, "Allow http connections (else forces https)")

	a := os.Args[1:]
	if args != nil {
		a = args
	}
	flag.CommandLine.Parse(a)

	return cliargs
}

func getHeaderPaths(b *bgzf.Reader) ([][]string, []string) {
	line, err := readLine(b)
	if err != nil {
		log.Fatalf("Error reading header line: [%s]\n", err)
	}

	headers := strings.Fields(string(line))

	headerPaths := [][]string{}

	for _, header := range headers {
		path := strings.Split(header, ".")

		headerPaths = append(headerPaths, path)
	}

	return headerPaths, headers
}

func createIndex(opensearchConnectionConfigPath string, opensearchIndexConfigPath string, indexName string, fileSize int64) (opensearch.Config, *opensearch.Client, OpensearchMappingConfig) {
	var osearchConnConfig OpensearchConnectionConfig
	var osearchMapConfig OpensearchMappingConfig

	connectionSettings, err := os.ReadFile(opensearchConnectionConfigPath)
	if err != nil {
		log.Fatalf("Couldn't read connection settings due to: [%s]\n", err)
	}

	err = yaml.Unmarshal(connectionSettings, &osearchConnConfig)
	if err != nil {
		log.Fatalf("Failed to unmarshal search configuration: [%s]\n", err)
	}

	indexConfig, err := os.ReadFile(opensearchIndexConfigPath)
	if err != nil {
		log.Fatalf("Couldn't read index configuration due to: [%s]\n", err)
	}

	err = yaml.Unmarshal(indexConfig, &osearchMapConfig)
	if err != nil {
		log.Fatalf("Unmarshal failed due to: [%s]\n", err)
	}

	_, ok := osearchMapConfig.Settings["index"]["number_of_shards"]

	if !ok {
		// No more than 10GB per shard
		osearchMapConfig.Settings["index"]["number_of_shards"] = math.Ceil(float64(fileSize) / float64(1e10))
	}

	settings := osearchMapConfig.Settings
	indexSettings, err := sonic.Marshal(settings)
	if err != nil {
		log.Fatalf("Marshaling failed of index settings due to: [%s]\n", err)
	}

	indexMapping, err := sonic.Marshal(osearchMapConfig.Mappings)
	if err != nil {
		log.Fatalf("Marshaling failed of mappings failed due to: [%s]\n", err)
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
		log.Fatalf("Error deleting index due to: [%s]\n", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		deleteIndex := opensearchapi.IndicesDeleteRequest{
			Index: []string{indexName},
		}

		deleteIndexResponse, err := deleteIndex.Do(context.Background(), client)

		if err != nil || deleteIndexResponse.IsError() {
			log.Fatalf("Error deleting index due to: [%s]\n", err)
		}
	}

	createIndex := opensearchapi.IndicesCreateRequest{
		Index: indexName,
		Body:  strings.NewReader(requestBody),
	}

	createResp, err := createIndex.Do(context.Background(), client)

	if err != nil || createResp.IsError() {
		log.Fatalf("Error creating index due to: [%s]\n", err)
	}

	defer createResp.Body.Close()

	// JSON body to update the index settings
	body := `{
        "index.mapping.total_fields.limit": 10000
    }`

	// Update index settings
	res, err := client.Indices.PutSettings(
		strings.NewReader(body),
		client.Indices.PutSettings.WithIndex(indexName), // Replace with your index name
	)
	if err != nil {
		panic(err)
	}
	defer res.Body.Close()

	return osConfig, client, osearchMapConfig
}

// Read a line up to the next newline character, and return the line excluding the newline character
func _readLine(r *bgzf.Reader) ([]byte, error) {
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

func readLine(r *bgzf.Reader) ([]byte, error) {
	tx := r.Begin()
	data, err := _readLine(r)
	tx.End()
	return data, err
}

func readLines(b *bgzf.Reader) ([]byte, error) {
	buf := make([]byte, 64*1024*8) // 8 bgzip blocks at a time

	tx := b.Begin()
	defer tx.End()

	bytesRead, err := b.Read(buf)

	if buf[bytesRead-1] != '\n' {
		remainder, err := _readLine(b)
		return append(buf[:bytesRead], remainder...), err
	}
	// last byte is newline
	return buf[:bytesRead-1], err
}

func TestUse(bConfig BeanstalkdConfig) {
	// c, err := beanstalk.Dial("tcp", "127.0.0.1:11300")
	// func Dial(network, addr string) (*Conn, error) {
	// 	return DialTimeout(network, addr, DefaultDialTimeout)
	// }

	// c := NewConn(mock(
	// 	"use foo\r\nput 0 0 0 5\r\nhello\r\n",
	// 	"USING foo\r\nINSERTED 1\r\n",
	// ))
	// tube := NewTube(c, "foo")
	// id, err := tube.Put([]byte("hello"), 0, 0, 0)
	// if err != nil {
	// 	t.Fatal(err)
	// }
	// if id != 1 {
	// 	t.Fatal("expected 1, got", id)
	// }
	// if err = c.Close(); err != nil {
	// 	t.Fatal(err)
	// }
}

func main() {
	cliargs := setup(nil)

	indexName := cliargs.indexName

	// From testing, performance seems maximized at 32 threads for a 4 vCPU AWS instance
	concurrency := runtime.NumCPU() * 8

	//Open the tar archive
	// archive, err := os.Open(cliargs.annotationTarballPath)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// defer archive.Close()

	// tarReader := tar.NewReader(archive)

	// var b *bgzf.Reader
	// for {
	// 	header, err := tarReader.Next()
	// 	if err != nil {
	// 		if err == io.EOF {
	// 			break // End of archive
	// 		}
	// 		log.Fatal(err) // Handle other errors
	// 	}

	// 	// TODO @akotlar 2023-11-24: Take the expected file name from the information submitted in the beanstalkd queue message
	// 	if strings.HasSuffix(header.Name, "annotation.tsv.gz") {
	// 		b, err = bgzf.NewReader(tarReader, 0)
	// 		if err != nil {
	// 			log.Fatal(err)
	// 		}
	// 		break
	// 	}
	// }

	// // TODO @akotlar 2023-11-24: Get the file size of just the file being indexed; though the tar archive is about the same size as the file, it's not exactly the same
	// fileStats, err := archive.Stat()
	// if err != nil {
	// 	log.Fatal(err)
	// }
	file, err := os.Open("/home/ubuntu/bystro/rust/index_annotation/all_chr1_phase3_shapeit2_mvncall_integrated_v5b_20130502_genotypes_vcf.annotation.500klines.tsv.gz")

	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	fileStats, err := file.Stat()
	if err != nil {
		log.Fatal(err)
	}

	b, err := bgzf.NewReader(file, 0)

	if err != nil {
		log.Fatal(err)
	}

	workQueue := make(chan parser.Job)
	complete := make(chan bool)

	headerPaths, _ := getHeaderPaths(b)

	if len(headerPaths) == 0 {
		log.Fatal("No header paths found")
	}

	osConfig, client, osearchMapConfig := createIndex(cliargs.osConnectionConfigPath, cliargs.osIndexConfigPath, cliargs.indexName, fileStats.Size())
	// client.
	// Spawn threads
	for i := 0; i < concurrency; i++ {
		go parser.Parse(headerPaths, indexName, osConfig, workQueue, complete, i)
	}

	// c, err := beanstalk.Dial("tcp", "127.0.0.1:11300")
	// id, err := c.Put([]byte("hello"), 1, 0, 120*time.Second)

	chunkStart := 0
	var lines []string

	// outFh := os.Stdout
	// writer := bufio.NewWriterSize(outFh, 48*1024*1024)

	// writer.WriteString(strings.Join(header, "\t"))

	for {
		rawLines, err := readLines(b)
		// line, err := readLine(b)

		if len(rawLines) > 0 {
			lines = strings.Split(string(rawLines), parser.LINE_DELIMITER)
		}

		if err != nil {
			if err == io.EOF {
				if len(lines) > 0 {
					workQueue <- parser.Job{Lines: lines, Start: chunkStart}
					chunkStart += len(lines)
					lines = nil
				}
				break
			}
			log.Fatal(err)
		}

		// lines = strings.Split(string(rawLines), parser.LINE_DELIMITER)
		// lines = append(lines, string(line))
		// go parser.ParseDirect(headerPaths, indexName, osConfig, lines, 0)
		// workQueue <- parser.Job{Lines: lines, Start: chunkStart}

		if len(lines) > 0 {
			workQueue <- parser.Job{Lines: lines, Start: chunkStart}
			// fmt.Println(strings.Join(lines, "\n"))

			chunkStart += len(lines)
			lines = nil
		}
	}

	// if len(lines) > 0 {
	// 	fmt.Println(strings.Join(lines, "\n"))

	// 	chunkStart += len(lines)
	// 	lines = nil
	// }

	// Indicate to all processing threads that no more work remains
	close(workQueue)

	// // Wait for everyone to finish.
	for i := 0; i < concurrency; i++ {
		<-complete
	}

	fmt.Printf("Indexed %d lines\n", chunkStart)

	postIndexSettings, err := sonic.Marshal(osearchMapConfig.PostIndexSettings)

	if err != nil {
		log.Fatalf("Marshaling failed of post index settings: [%s]\n", err)
	}

	postIndexRequestBody := fmt.Sprintf(`{
		"settings": %s
	}`, string(postIndexSettings))

	req := opensearchapi.IndicesPutSettingsRequest{
		Index: []string{indexName},
		Body:  strings.NewReader(postIndexRequestBody),
	}

	res, err := req.Do(context.Background(), client)
	if err != nil {
		log.Fatalf("Error updating index settings: [%s]\n", err)
	}
	defer res.Body.Close()

	res, err = client.Indices.Flush(
		client.Indices.Flush.WithIndex(indexName),
	)
	if err != nil || res.IsError() {
		log.Fatal(err, res.StatusCode)
	}

	refreshRes, err := client.Indices.Refresh(client.Indices.Refresh.WithIndex(indexName))
	if err != nil || refreshRes.IsError() {
		log.Fatalf("Error refreshing index: [%s]\n", err.Error())
	}

	refreshRes.Body.Close()
}
