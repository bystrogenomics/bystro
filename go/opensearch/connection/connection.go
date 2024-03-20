package connection

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"

	"github.com/bytedance/sonic"
	opensearch "github.com/opensearch-project/opensearch-go/v2"
	opensearchapi "github.com/opensearch-project/opensearch-go/v2/opensearchapi"
	"gopkg.in/yaml.v3"
)

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

func CreateIndex(opensearchConnectionConfigPath string, opensearchIndexConfigPath string, indexName string, fileSize int64) (opensearch.Config, *opensearch.Client, OpensearchMappingConfig) {
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
		Username:      osearchConnConfig.Auth.Username,
		Password:      osearchConnConfig.Auth.Password,
		MaxRetries:    5,
		RetryOnStatus: []int{502, 503, 504},
	}

	client, err := opensearch.NewClient(osConfig)

	if err != nil {
		log.Fatalf("Error creating OpenSearch client due to: [%s]\n", err)
	}

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

func CompleteIndexRequest(client *opensearch.Client, osearchMapConfig OpensearchMappingConfig, indexName string) error {
	postIndexSettings, err := sonic.Marshal(osearchMapConfig.PostIndexSettings)
	if err != nil {
		return err
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
		return err
	}

	defer res.Body.Close()

	res, err = client.Indices.Flush(
		client.Indices.Flush.WithIndex(indexName),
	)
	if err != nil {
		return err
	}
	if res.IsError() {
		return fmt.Errorf("error flushing index: %s", res)
	}

	refreshRes, err := client.Indices.Refresh(client.Indices.Refresh.WithIndex(indexName))
	if err != nil {
		return err
	}
	if refreshRes.IsError() {
		return fmt.Errorf("error refreshing index: %s", refreshRes)
	}

	refreshRes.Body.Close()

	return err
}
