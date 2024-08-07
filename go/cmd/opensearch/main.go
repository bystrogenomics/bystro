package main

import (
	"bystro/beanstalkd"
	"bystro/decompress"
	"bystro/opensearch/connection"
	"bystro/opensearch/parser"
	"flag"
	"runtime"

	"fmt"
	"io"
	"log"

	"os"
	"strings"

	"github.com/bytedance/sonic"
)

type CLIArgs struct {
	annotationPath         string
	osIndexConfigPath      string
	beanstalkConfigPath    string
	osConnectionConfigPath string
	indexName              string
	jobSubmissionID        string
	progressFrequency      int
}

func setup(args []string) *CLIArgs {
	cliargs := &CLIArgs{}
	flag.StringVar(&cliargs.annotationPath, "input", "", "The path to the input annotation")
	flag.StringVar(&cliargs.annotationPath, "in", "", "The path to the input annotation (short form)")
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
	if v.annotationPath == "" {
		missing = append(missing, "input")
	}
	if v.osIndexConfigPath == "" {
		missing = append(missing, "os-index-config-path")
	}
	if v.osConnectionConfigPath == "" {
		missing = append(missing, "os-connection-config-path")
	}
	if v.indexName == "" {
		missing = append(missing, "index-name")
	}
	if v.jobSubmissionID == "" {
		if v.beanstalkConfigPath != "" {
			missing = append(missing, "job-submission-id")
		}
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required arguments: %s", strings.Join(missing, ", "))
	}
	return nil
}

func main() {
	cliargs := setup(nil)

	indexName := cliargs.indexName

	// From testing, performance seems maximized at 32 threads for a 4 vCPU AWS instance
	concurrency := runtime.NumCPU() * 8

	workQueue := make(chan parser.Job)
	complete := make(chan bool)

	input, err := os.Open(cliargs.annotationPath)
	if err != nil {
		log.Fatalf("Couldn't open annotation due to: [%s]\n", err)
	}
	defer input.Close()

	reader, err := decompress.GetAnnotationFh(input)
	if err != nil {
		log.Fatalf("Couldn't open annotation file due to: [%s]\n", err)
	}

	// Get the file size
	fileStats, err := input.Stat()
	if err != nil {
		log.Fatalf("Couldn't get file stats due to: [%s]\n", err)
	}

	headerPaths, headerFields := decompress.GetHeaderPaths(reader)

	if len(headerPaths) == 0 {
		log.Fatal("No header found")
	}

	osConfig, client, osearchMapConfig := connection.CreateIndex(cliargs.osConnectionConfigPath, cliargs.osIndexConfigPath, cliargs.indexName, fileStats.Size())

	// Spawn threads
	for i := 0; i < concurrency; i++ {
		go parser.Parse(headerPaths, indexName, osConfig, workQueue, complete, i)
	}

	progressSender, err := beanstalkd.CreateMessageSender(cliargs.beanstalkConfigPath, cliargs.jobSubmissionID, "index")
	if err != nil {
		log.Fatalf("Couldn't create message sender due to: [%s]\n", err)
	}

	progressUpdate := 0
	chunkStart := 0
	var lines []string

	for {
		rawLines, err := reader.ReadLines()

		if len(rawLines) > 0 {
			lines = strings.Split(string(rawLines), parser.LINE_DELIMITER)
		}

		if err != nil {
			if err == io.EOF {
				if len(lines) > 0 {
					workQueue <- parser.Job{Lines: lines, Start: chunkStart}
					chunkStart += len(lines)
					lines = nil

					progressSender.SetProgress(chunkStart)
					progressSender.SendMessage()
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
			progressSender.SetProgress(chunkStart)
			go progressSender.SendMessage()
			progressUpdate = 0
		}
	}

	// Indicate to all processing threads that no more work remains
	close(workQueue)

	// // Wait for everyone to finish.
	for i := 0; i < concurrency; i++ {
		<-complete
	}

	progressSender.SetProgress(chunkStart)
	progressSender.SendMessage()

	err = progressSender.Close()
	if err != nil {
		log.Printf("Error closing progress sender due to: [%s]\n", err)
	}

	err = connection.CompleteIndexRequest(client, osearchMapConfig, indexName)
	if err != nil {
		log.Fatalf("Error completing index request due to: [%s]\n", err)
	}

	// Return JSON of header fields and progressSender.Data.Progress and progressSender.Data.Skipped
	returnMsgJson, err := sonic.Marshal(
		map[string]interface{}{
			"header":       headerFields,
			"totalIndexed": progressSender.GetProgress(),
		},
	)
	if err != nil {
		// Handle error
		log.Fatalf("Error marshaling JSON: %v", err)
	}
	fmt.Print(string(returnMsgJson))
}
