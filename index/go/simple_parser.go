package main

import (
	"bufio"
	"context"
	"crypto/tls"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"

	"math"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"encoding/json"

	opensearch "github.com/opensearch-project/opensearch-go/v2"
	opensearchapi "github.com/opensearch-project/opensearch-go/v2/opensearchapi"
	opensearchutil "github.com/opensearch-project/opensearch-go/v2/opensearchutil"
	"gopkg.in/yaml.v3"
	// "unicode"
)

func FindEndOfLine(r *bufio.Reader, s string) (byte, int, string, error) {
	runeChar, _, err := r.ReadRune()

	if err != nil {
		return byte(0), 0, "", err
	}

	if runeChar == '\r' {
		nextByte, err := r.Peek(1)

		if err != nil {
			return byte(0), 0, "", err
		}

		if rune(nextByte[0]) == '\n' {
			_, _, err = r.ReadRune()

			if err != nil {
				return byte(0), 0, "", err
			}

			return nextByte[0], 2, s, nil
		}

		return byte('\r'), 1, s, nil
	}

	if runeChar == '\n' {
		return byte('\n'), 1, s, nil
	}

	s += string(runeChar)
	return FindEndOfLine(r, s)
}

func populateHashPath2(rowDocument map[string]interface{}, fieldPath []string, fieldValue interface{}) map[string]interface{} {
	current := rowDocument
	var key string
	for i := 0; i < len(fieldPath); i += 1 {
		key = fieldPath[i]
		if _, ok := current[key]; !ok {
			current[key] = make(map[string]interface{})
		}
		if i < len(fieldPath)-1 {
			current = current[key].(map[string]interface{})
		} else {
			current[key] = fieldValue
		}
	}

	return rowDocument
}

type CLIArgs struct {
	annotationTarballPath  string
	osIndexConfigPath      string
	osConnectionConfigPath string
	indexName              string
	allowHTTP              bool
}

type OpensearchConnectionConfig struct {
	Connection struct {
		MaxContentLength int      `yaml:"max_content_length"`
		RequestTimeout   int      `yaml:"request_timeout"`
		Gzip             bool     `yaml:"gzip"`
		Username         string   `yaml:"username"`
		Password         string   `yaml:"password"`
		Nodes            []string `yaml:"nodes"`
	} `yaml:"connection"`
}

type OpensearchMappingConfig struct {
	NumericalFields   []string                          `yaml:"numericalFields" json:"numericalFields"`
	Sort              map[string]string                 `yaml:"sort" json:"sort"`
	BooleanFields     []string                          `yaml:"booleanFields" json:"booleanFields"`
	PostIndexSettings map[string]map[string]interface{} `yaml:"post_index_settings" json:"post_index_settings"`
	Settings          map[string]map[string]interface{} `yaml:"index_settings" json:"index_settings"`
	Mappings          map[string]map[string]interface{} `yaml:"mappings" json:"mappings"`
}

func setup(args []string) *CLIArgs {
	cliargs := &CLIArgs{}
	flag.StringVar(&cliargs.annotationTarballPath, "in", "example.tar", "The path to the input tarball")
	flag.StringVar(&cliargs.osIndexConfigPath, "index", "test_hg19.mapping.yml", "The path to the OpenSearch mapping and index definition (e.g. hg19.mapping.yml)")
	flag.StringVar(&cliargs.osConnectionConfigPath, "connection", "test_opensearch.connection.yml", "The path to the OpenSearch connection config (e.g. config/elasticsearch.yml)")
	flag.StringVar(&cliargs.indexName, "name", "test2", "The index name")
	flag.BoolVar(&cliargs.allowHTTP, "http", false, "Allow http connections (else forces https)")

	a := os.Args[1:]
	if args != nil {
		a = args
	}
	flag.CommandLine.Parse(a)

	return cliargs
}

func main() {
	cliargs := setup(nil)

	var osearchConnConfig OpensearchConnectionConfig
	var osearchMapConfig OpensearchMappingConfig
	res, err := ioutil.ReadFile(cliargs.osConnectionConfigPath)
	if err != nil {
		log.Fatalf("Coudln't read: %s due to: %s", cliargs.osConnectionConfigPath, err)
	}

	err = yaml.Unmarshal(res, &osearchConnConfig)
	if err != nil {
		log.Fatalf("Unmarshal failed: %v", err)
	}

	index_config, err := ioutil.ReadFile(cliargs.osIndexConfigPath)
	if err != nil {
		log.Fatalf("Coudln't read: %s due to: %s", cliargs.osConnectionConfigPath, err)
	}

	err = yaml.Unmarshal(index_config, &osearchMapConfig)
	if err != nil {
		log.Fatalf("Unmarshal failed: %v", err)
	}

	client, err := opensearch.NewClient(opensearch.Config{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: cliargs.allowHTTP},
		},
		RetryOnStatus: []int{502, 503, 504, 429},
		RetryBackoff:  func(i int) time.Duration { return time.Duration(i) * 100 * time.Millisecond },
		MaxRetries:    5,
		Addresses:     osearchConnConfig.Connection.Nodes,
		Username:      osearchConnConfig.Connection.Username,
		Password:      osearchConnConfig.Connection.Password,
	})

	if err != nil {
		log.Fatal(err)
	}

	resp, err := opensearchapi.IndicesExistsRequest{
		Index: []string{cliargs.indexName},
	}.Do(context.Background(), client)

	if err != nil {
		log.Fatal(err)
	}

	if resp.StatusCode == 404 {
		fileStats, err := os.Stat(cliargs.annotationTarballPath)
		if err != nil {
			log.Fatal(err)
		}

		_, ok := osearchMapConfig.Settings["index"]["number_of_shards"]

		if !ok {
			osearchMapConfig.Settings["index"]["number_of_shards"] = math.Ceil(float64(fileStats.Size()) / float64(1e10))
		}

		indexSettings, err := json.Marshal(osearchMapConfig.Settings)
		if err != nil {
			log.Fatalf("JSON Marshalling failed: %v", err)
		}

		indexMapping, err := json.Marshal(osearchMapConfig.Mappings)
		if err != nil {
			log.Fatalf("JSON Marshalling failed: %v", err)
		}

		requestBody := fmt.Sprintf(`{
			"settings": %s,
			"mappings": %s
		}`, string(indexSettings), string(indexMapping))

		createIndex := opensearchapi.IndicesCreateRequest{
			Index: cliargs.indexName,
			Body:  strings.NewReader(requestBody),
		}

		createIndexResponse, err := createIndex.Do(context.Background(), client)

		if err != nil {
			log.Fatalf("failed to create index %v", err)
		}

		if createIndexResponse.StatusCode != 200 {
			deleteIndex := opensearchapi.IndicesDeleteRequest{
				Index: []string{cliargs.indexName},
			}
			log.Printf("Deleted index %s: %v", cliargs.indexName, deleteIndex)
			log.Fatalf("failed to create index %v", err)
		}

		log.Println(createIndexResponse, err)
	}

	indexer, err := opensearchutil.NewBulkIndexer(opensearchutil.BulkIndexerConfig{
		Client:     client,
		Index:      cliargs.indexName,
		NumWorkers: runtime.NumCPU(),
		FlushBytes: 5e+7,
		OnError: func(ctx context.Context, err error) {
			log.Printf("Indexing error: %v", err)
		},
	})

	if err != nil {
		log.Fatalf("Error creating the indexer: %s", err)
	}

	// TODO: read these from genome assembly config (e.g. hg19.yml)
	alleleDelimiter := "/"
	positionDelimiter := "|"
	valueDelimiter := ";"
	emptyFieldChar := "!"
	fieldSeparator := "\t"

	res, err = exec.Command("bash", "-c", fmt.Sprintf("tar -tf %s | grep annotation.tsv.gz", cliargs.annotationTarballPath)).Output()
	if err != nil {
		log.Fatal(err)
	}

	annotationPath := strings.TrimRight(string(res), "\r\n")

	extractCmd := fmt.Sprintf("tar -O -xf %s %s | gzip -dc", cliargs.annotationTarballPath, annotationPath)

	cmd := exec.Command("bash", "-c", extractCmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}

	defer stdout.Close()

	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}

	reader := bufio.NewReaderSize(stdout, 48*1024*1024)
	endOfLineByte, numChars, headerRow, err := FindEndOfLine(reader, "")
	if err != nil {
		log.Fatal(err)
	}

	var paths [][]string
	headerFields := strings.Split(headerRow, fieldSeparator)
	for _, field := range headerFields {
		paths = append(paths, strings.Split(field, "."))
	}

	booleanMap := make(map[string]bool)
	for _, field := range osearchMapConfig.BooleanFields {
		booleanMap[field] = true
	}

	var rowIdx int
	var rowDocumentJson []byte
	for {
		// http://stackoverflow.com/questions/8757389/reading-file-line-by-line-in-go
		// http://www.jeffduckett.com/blog/551119d6c6b86364cef12da7/golang---read-a-file-line-by-line.html
		rowStr, err := reader.ReadString(endOfLineByte) // 0x0A separator = newline

		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		} else if rowStr == "" {
			continue
		}

		// Chomp equivalent: https://groups.google.com/forum/#!topic/golang-nuts/smFU8TytFr4
		row := strings.Split(rowStr[:len(rowStr)-numChars], fieldSeparator)

		rowDocument := make(map[string]interface{})

		for i, field := range row {
			var alleleValues []interface{}
			for _, alleleValue := range strings.Split(field, alleleDelimiter) {
				if alleleValue == emptyFieldChar {
					alleleValues = append(alleleValues, nil)
					continue
				}

				var positionValues []interface{}
				for _, posValue := range strings.Split(alleleValue, positionDelimiter) {
					if posValue == emptyFieldChar {
						positionValues = append(positionValues, nil)
						continue
					}

					var values []interface{}
					values_raw := strings.Split(posValue, valueDelimiter)
					for _, value := range values_raw {
						if value == emptyFieldChar {
							values = append(values, nil)
							continue
						}

						if _, ok := booleanMap[headerFields[i]]; ok {
							if value == "1" {
								values = append(values, true)
							} else if value == "0" {
								values = append(values, false)
							} else {
								log.Fatalf("Encountered boolean value that wasn't encoded as 0/1 in field %s, row %d, value %s", field, i, value)
							}
						} else {
							values = append(values, value)
						}
					}

					if len(values_raw) > 1 {
						positionValues = append(positionValues, values)
					} else {
						positionValues = append(positionValues, values[0])
					}
				}

				alleleValues = append(alleleValues, positionValues)

			}

			rowDocument = populateHashPath2(rowDocument, paths[i], alleleValues)
		}

		rowDocumentJson, err = json.Marshal(rowDocument)
		if err != nil {
			log.Fatal(err)
		}

		err = indexer.Add(
			context.Background(),
			opensearchutil.BulkIndexerItem{
				Action:     "index",
				DocumentID: strconv.Itoa(rowIdx),
				Body:       strings.NewReader(string(rowDocumentJson)),
				OnFailure: func(
					ctx context.Context,
					item opensearchutil.BulkIndexerItem,
					res opensearchutil.BulkIndexerResponseItem, err error,
				) {
					if err != nil {
						log.Printf("ERROR: %s", err)
					} else {
						log.Printf("ERROR: %s: %s", res.Error.Type, res.Error.Reason)
					}
				},
			},
		)

		if err != nil {
			log.Fatalf("Unexpected error: %s", err)
		}

		rowIdx += 1
	}

	if err := cmd.Wait(); err != nil {
		log.Fatal(err)
	}

	if err := indexer.Close(context.Background()); err != nil {
		log.Fatalf("Unexpected error: %s", err)
	}

	stats := indexer.Stats()
	if stats.NumFailed > 0 {
		log.Fatalf("Indexed [%d] documents with [%d] errors", stats.NumFlushed, stats.NumFailed)
	} else {
		log.Printf("Successfully indexed [%d] documents", stats.NumFlushed)
	}

	postIndexSettingsJson, err := json.Marshal(osearchMapConfig.PostIndexSettings)
	if err != nil {
		log.Fatalf("JSON Marshalling failed: %v", err)
	}

	_, err = opensearchapi.IndicesPutSettingsRequest{
		Index: []string{cliargs.indexName},
		Body:  strings.NewReader(string(postIndexSettingsJson)),
	}.Do(context.Background(), client)

	if err != nil {
		log.Fatal(err)
	}

	ret := map[string]interface{}{
		"fieldNames":  headerFields,
		"indexConfig": osearchMapConfig,
	}

	res, err = json.Marshal(ret)

	if err != nil {
		log.Fatal(err)
	}

	os.Stdout.Write(res)
}
