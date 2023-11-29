package main

import (
	"archive/tar"
	"bufio"
	"bystro/pkg/parser"
	"context"
	"crypto/tls"
	"flag"
	"io/fs"
	"math"
	"net/http"
	"runtime"

	"fmt"
	"io"
	"log"

	"os"
	"strings"

	"github.com/beanstalkd/go-beanstalk"
	"github.com/biogo/hts/bgzf"
	"github.com/bytedance/sonic"
	gzip "github.com/klauspost/pgzip"
	opensearch "github.com/opensearch-project/opensearch-go/v2"
	opensearchapi "github.com/opensearch-project/opensearch-go/v2/opensearchapi"
	"gopkg.in/yaml.v3"
)

const EXPECTED_ANNOTATION_FILE_SUFFIX = "annotation.tsv.gz"
const PROGRESS_EVENT = "progress"

type BystroReader interface {
	readLines() ([]byte, error)
	readLine() ([]byte, error)
}

type BzfBystroReader struct {
	Reader *bgzf.Reader
}

type BufioBystroReader struct {
	Reader *bufio.Reader
}

// Read a line up to the next newline character, and return the line excluding the newline character
func _readLineBgzip(r *bgzf.Reader) ([]byte, error) {
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
	data, err := _readLineBgzip(r)
	tx.End()
	return data, err
}

func readLinesBgzip(r *bgzf.Reader) ([]byte, error) {
	buf := make([]byte, 64*1024*8) // 8 bgzip blocks at a time

	tx := r.Begin()
	defer tx.End()

	bytesRead, err := r.Read(buf)

	if bytesRead == 0 {
		return nil, err
	}

	if buf[bytesRead-1] != '\n' {
		remainder, err := _readLineBgzip(r)
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

func readLines(r *bufio.Reader) ([]byte, error) {
	buf := make([]byte, 64*1024*8) // 8 bgzip blocks at a time

	bytesRead, err := r.Read(buf)

	if bytesRead == 0 {
		return nil, err
	}

	if buf[bytesRead-1] != '\n' {
		remainder, err := readLine(r)
		return append(buf[:bytesRead], remainder...), err
	}
	// last byte is newline
	return buf[:bytesRead-1], err
}

func (r BzfBystroReader) readLines() ([]byte, error) {
	return readLinesBgzip(r.Reader)
}

func (r BzfBystroReader) readLine() ([]byte, error) {
	return readLineBgzip(r.Reader)
}

func (r BufioBystroReader) readLines() ([]byte, error) {
	return readLines(r.Reader)
}
func (r BufioBystroReader) readLine() ([]byte, error) {
	return r.Reader.ReadBytes('\n')
}

type CLIArgs struct {
	annotationTarballPath  string
	osIndexConfigPath      string
	beanstalkConfigPath    string
	osConnectionConfigPath string
	indexName              string
	jobSubmissionID        string
	noBeanstalkd           bool
	progressFrequency      int
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
	Progress int `json:"progress"`
	Skipped  int `json:"skipped"`
}

type ProgressMessage struct {
	SubmissionID string       `json:"submissionID"`
	Data         ProgressData `json:"data"`
	Event        string       `json:"event"`
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

type BeanstalkdYAML struct {
	Beanstalkd BeanstalkdConfig `yaml:"beanstalkd"`
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
	flag.StringVar(&cliargs.annotationTarballPath, "tarball-path", "", "The path to the input tarball")
	flag.StringVar(&cliargs.annotationTarballPath, "t", "", "The path to the input tarball (short form)")
	flag.StringVar(&cliargs.osIndexConfigPath, "mapping-config", "", "The path to the OpenSearch mapping and index definition (e.g. hg19.mapping.yml)")
	flag.StringVar(&cliargs.osIndexConfigPath, "m", "", "The path to the OpenSearch mapping and index definition (short form)")
	flag.StringVar(&cliargs.osConnectionConfigPath, "opensearch-config", "", "The path to the OpenSearch connection config (e.g. config/elasticsearch.yml)")
	flag.StringVar(&cliargs.osConnectionConfigPath, "o", "", "The path to the OpenSearch connection config (short form)")
	flag.StringVar(&cliargs.beanstalkConfigPath, "queue-config", "", "The path to the Beanstalkd queue connection config (e.g. config/beanstalk.yml)")
	flag.StringVar(&cliargs.beanstalkConfigPath, "q", "", "The path to the Beanstalkd queue connection config (short form)")
	flag.StringVar(&cliargs.indexName, "index-name", "", "The index name")
	flag.StringVar(&cliargs.indexName, "i", "", "The index name (short form)")
	flag.StringVar(&cliargs.jobSubmissionID, "job-submission-id", "", "The job submission ID")
	flag.StringVar(&cliargs.jobSubmissionID, "j", "", "The job submission ID (short form)")
	flag.BoolVar(&cliargs.noBeanstalkd, "no-queue", false, "Disable beanstalkd progress events")
	flag.BoolVar(&cliargs.noBeanstalkd, "n", false, "Disable beanstalkd progress events (short form)")
	flag.IntVar(&cliargs.progressFrequency, "progress-frequency", 5e3, "Print progress every N variants processed")
	flag.IntVar(&cliargs.progressFrequency, "p", 5e3, "Print progress every N variants processed (short form)")

	a := os.Args[1:]
	if args != nil {
		a = args
	}

	flag.CommandLine.Parse(a)

	if err := validateArgs(cliargs); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		flag.Usage()
		os.Exit(1)
	}

	return cliargs
}

func validateArgs(args *CLIArgs) error {
	var missing []string
	v := *args
	if v.annotationTarballPath == "" {
		missing = append(missing, "annotation-tarball-path")
	}
	if v.osIndexConfigPath == "" {
		missing = append(missing, "os-index-config-path")
	}
	if v.osConnectionConfigPath == "" {
		missing = append(missing, "os-connection-config-path")
	}
	if v.beanstalkConfigPath == "" {
		if !v.noBeanstalkd {
			missing = append(missing, "beanstalk-config-path")
		}
	}
	if v.indexName == "" {
		missing = append(missing, "index-name")
	}
	if v.jobSubmissionID == "" {
		missing = append(missing, "job-submission-id")
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required arguments: %s", strings.Join(missing, ", "))
	}
	return nil
}

func getHeaderPaths(b BystroReader) ([][]string, []string) {
	line, err := b.readLine()
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
		createRespBody, err := io.ReadAll(createResp.Body)
		if err != nil {
			log.Fatalf("Error reading the response body due to: %s", err)
		}

		log.Fatalf("Error creating index due to: [%s], status: [%s], response: [%s]\n", err, createResp.Status(), createRespBody)
	}

	createResp.Body.Close()

	return osConfig, client, osearchMapConfig
}

func createBeanstalkdConfig(beanstalkConfigPath string) (BeanstalkdConfig, error) {
	var bConfig BeanstalkdYAML

	beanstalkConfig, err := os.ReadFile(beanstalkConfigPath)
	if err != nil {
		return BeanstalkdConfig{}, err
	}

	err = yaml.Unmarshal(beanstalkConfig, &bConfig)
	if err != nil {
		return BeanstalkdConfig{}, err
	}

	return bConfig.Beanstalkd, nil
}

func _getAnnotationFhFromTarArchive(archive *os.File) (*tar.Reader, fs.FileInfo, error) {
	fileStats, err := archive.Stat()
	if err != nil {
		return nil, nil, err
	}

	tarReader := tar.NewReader(archive)

	for {
		header, err := tarReader.Next()
		if err != nil {
			if err == io.EOF {
				break // End of archive
			}
			return nil, nil, err
		}

		// TODO @akotlar 2023-11-24: Take the expected file name from the information submitted in the beanstalkd queue message
		if strings.HasSuffix(header.Name, EXPECTED_ANNOTATION_FILE_SUFFIX) {
			return tarReader, fileStats, nil
		}
	}

	return nil, nil, fmt.Errorf("couldn't find annotation file in tarball")
}

func getAnnotationFhFromTarArchive(archive *os.File) (BystroReader, fs.FileInfo, error) {
	tarReader, fileStats, err := _getAnnotationFhFromTarArchive(archive)

	if err != nil {
		return nil, nil, err
	}

	b, err := bgzf.NewReader(tarReader, 0)
	if err != nil {
		archive.Seek(0, 0)

		tarReader, fileStats, err := _getAnnotationFhFromTarArchive(archive)

		if err != nil {
			return nil, nil, err
		}

		b, err := gzip.NewReader(tarReader)

		if err != nil {
			return nil, nil, err
		}

		bufioReader := bufio.NewReader(b)

		return BufioBystroReader{Reader: bufioReader}, fileStats, nil
	}

	return BzfBystroReader{Reader: b}, fileStats, err
}

func sendEvent(message ProgressMessage, eventTube *beanstalk.Tube, noBeanstalkd bool) {
	if noBeanstalkd {
		fmt.Printf("Indexed %d annotated variants\n", message.Data.Progress)
		return
	}

	messageJson, err := sonic.Marshal(message)
	if err != nil {
		log.Fatalf("Marshaling failed of progress message: [%s]\n", err)
	}

	eventTube.Put(messageJson, 0, 0, 0)
}

func main() {
	cliargs := setup(nil)

	indexName := cliargs.indexName

	// From testing, performance seems maximized at 32 threads for a 4 vCPU AWS instance
	concurrency := runtime.NumCPU() * 8

	workQueue := make(chan parser.Job)
	complete := make(chan bool)

	archive, err := os.Open(cliargs.annotationTarballPath)
	if err != nil {
		log.Fatalf("Couldn't open tarball due to: [%s]\n", err)
	}
	defer archive.Close()

	reader, fileStats, err := getAnnotationFhFromTarArchive(archive)

	if err != nil {
		log.Fatalf("Couldn't get annotation file handle due to: [%s]\n", err)
	}

	headerPaths, headerFields := getHeaderPaths(reader)

	if len(headerPaths) == 0 {
		log.Fatal("No header found")
	}

	osConfig, client, osearchMapConfig := createIndex(cliargs.osConnectionConfigPath, cliargs.osIndexConfigPath, cliargs.indexName, fileStats.Size())

	// Spawn threads
	for i := 0; i < concurrency; i++ {
		go parser.Parse(headerPaths, indexName, osConfig, workQueue, complete, i)
	}

	var beanstalkdConfig BeanstalkdConfig
	var eventTube *beanstalk.Tube
	var message ProgressMessage

	if !cliargs.noBeanstalkd {
		beanstalkdConfig, err = createBeanstalkdConfig(cliargs.beanstalkConfigPath)
		if err != nil {
			log.Fatalf("Couldn't create beanstalkd config due to: [%s]\n", err)
		}
		beanstalkConnection, err := beanstalk.Dial("tcp", beanstalkdConfig.Addresses[0])
		if err != nil {
			log.Fatalf("Couldn't connect to beanstalkd due to: [%s]\n", err)
		}
		defer beanstalkConnection.Close()
		eventTube = beanstalk.NewTube(beanstalkConnection, beanstalkdConfig.Tubes.Index.Events)

		message = ProgressMessage{
			SubmissionID: cliargs.jobSubmissionID,
			Event:        PROGRESS_EVENT,
			Data: ProgressData{
				Progress: 0,
				Skipped:  0,
			},
		}
	}

	progressUpdate := 0
	chunkStart := 0
	var lines []string

	for {
		rawLines, err := reader.readLines()

		if len(rawLines) > 0 {
			lines = strings.Split(string(rawLines), parser.LINE_DELIMITER)
		}

		if err != nil {
			if err == io.EOF {
				if len(lines) > 0 {
					workQueue <- parser.Job{Lines: lines, Start: chunkStart}
					chunkStart += len(lines)
					lines = nil

					message.Data.Progress = chunkStart
					sendEvent(message, eventTube, cliargs.noBeanstalkd)
				}
				break
			}
			log.Fatalf("Error reading lines: [%s]\n", err)
		}

		if len(lines) > 0 {
			workQueue <- parser.Job{Lines: lines, Start: chunkStart}

			chunkStart += len(lines)
			progressUpdate += len(lines)
			lines = nil
		}

		if progressUpdate >= cliargs.progressFrequency {
			message.Data.Progress = chunkStart
			go sendEvent(message, eventTube, cliargs.noBeanstalkd)
			progressUpdate = 0
		}
	}

	// Indicate to all processing threads that no more work remains
	close(workQueue)

	// // Wait for everyone to finish.
	for i := 0; i < concurrency; i++ {
		<-complete
	}

	message.Data.Progress = chunkStart
	sendEvent(message, eventTube, cliargs.noBeanstalkd)

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
		log.Fatalf("Error refreshing index: [%s]\n", err)
	}

	refreshRes.Body.Close()

	marshalledHeaderFields, err := sonic.Marshal(headerFields)
	if err != nil {
		log.Fatalf("Marshaling failed of header fields: [%s]\n", err)
	}

	fmt.Print(string(marshalledHeaderFields))
}
