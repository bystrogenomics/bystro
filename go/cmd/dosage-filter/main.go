package main

import (
	"bufio"
	"bystro/beanstalkd"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"sync/atomic"

	"github.com/apache/arrow/go/v14/arrow/array"
	"github.com/apache/arrow/go/v14/arrow/ipc"
	"github.com/apache/arrow/go/v14/arrow/memory"
	bystroArrow "github.com/bystrogenomics/bystro-vcf/arrow"
)

var WRITE_CHUNK_SIZE = 1000

type CLIArgs struct {
	inputPath           string
	outputPath          string
	beanstalkConfigPath string
	jobSubmissionID     string
	lociPath            string
	noBeanstalkd        bool
	progressFrequency   int64
}

// setup parses the command-line arguments and returns a CLIArgs struct.
func setup(args []string) *CLIArgs {
	cliargs := &CLIArgs{}
	flag.StringVar(&cliargs.inputPath, "input", "", "The path to the input Arrow IPC/Feather file")
	flag.StringVar(&cliargs.inputPath, "in", "", "The path to the input Arrow IPC/Feather file")

	flag.StringVar(&cliargs.lociPath, "loci", "", "The path to a file containing loci to filter, one locus per line")

	flag.StringVar(&cliargs.outputPath, "output", "", "The path to the output Arrow IPC/Feather file")
	flag.StringVar(&cliargs.outputPath, "out", "", "The path to the output Arrow IPC/Feather file")

	flag.StringVar(&cliargs.beanstalkConfigPath, "queue-config", "", "The path to the Beanstalkd queue connection config (e.g. config/beanstalk.yml)")
	flag.StringVar(&cliargs.beanstalkConfigPath, "q", "", "The path to the Beanstalkd queue connection config (short form)")
	flag.BoolVar(&cliargs.noBeanstalkd, "no-queue", true, "Disable beanstalkd progress events")
	flag.BoolVar(&cliargs.noBeanstalkd, "n", true, "Disable beanstalkd progress events (short form)")
	flag.Int64Var(&cliargs.progressFrequency, "progress-frequency", int64(5e3), "Print progress every N variants processed")
	flag.Int64Var(&cliargs.progressFrequency, "p", int64(5e3), "Print progress every N variants processed (short form)")

	flag.StringVar(&cliargs.jobSubmissionID, "job-submission-id", "", "The job submission ID")
	flag.StringVar(&cliargs.jobSubmissionID, "j", "", "The job submission ID (short form)")

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

// validateArgs checks if the required arguments are provided.
func validateArgs(args *CLIArgs) error {
	var missing []string
	v := *args
	if v.inputPath == "" {
		missing = append(missing, "input")
	}
	if v.outputPath == "" {
		missing = append(missing, "output")
	}
	if v.lociPath == "" {
		missing = append(missing, "loci")
	}
	if v.beanstalkConfigPath == "" {
		if !v.noBeanstalkd {
			missing = append(missing, "beanstalk-config-path")
		}
	}

	if v.jobSubmissionID == "" {
		if !v.noBeanstalkd {
			missing = append(missing, "job-submission-id")
		}
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required arguments: %s", strings.Join(missing, ", "))
	}

	return nil
}

// readLociFile reads a file containing loci to filter and returns a map of loci.
// Each line in the file represents a locus.
func readLociFile(filePath string) (map[string]bool, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	loci := make(map[string]bool)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		loci[scanner.Text()] = true
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return loci, nil
}

// processRecordAt processes a record at the given index, filters the rows based on the loci,
// and writes the filtered rows to the arrowWriter.
func processRecordAt(fr *ipc.FileReader, loci map[string]bool, arrowWriter *bystroArrow.ArrowWriter, queue chan int, complete chan bool, count *atomic.Int64) {
	pool := memory.NewGoAllocator()
	builder := array.NewRecordBuilder(pool, arrowWriter.Schema)
	defer builder.Release()

	rowsAccumulated := 0

	totalRows := int64(len(loci))
	for index := range queue {
		// We don't have an easy way to close the channel
		// so we'll just quickly skip to the end
		if count.Load() >= totalRows {
			continue
		}

		record, err := fr.RecordAt(index)
		if err != nil {
			log.Fatalf("Failed to read record at %d: %v", index, err)
		}

		locusCol := record.Column(0).(*array.String)

		rowsAccepted := 0
		for j := 0; j < int(record.NumRows()); j++ {
			locusValue := locusCol.Value(j)
			if _, exists := loci[locusValue]; exists {
				rowsAccumulated += 1
				rowsAccepted += 1
				// Fill the builder with values from the original columns, filtering by selectedIndices
				for colIdx := 0; colIdx < int(record.NumCols()); colIdx++ {
					column := record.Column(colIdx)
					switch column := column.(type) {
					case *array.String:
						builder.Field(colIdx).(*array.StringBuilder).Append(column.Value(j))
					case *array.Uint8:
						builder.Field(colIdx).(*array.Uint8Builder).Append(column.Value(j))
					case *array.Uint16:
						builder.Field(colIdx).(*array.Uint16Builder).Append(column.Value(j))
					case *array.Uint32:
						builder.Field(colIdx).(*array.Uint32Builder).Append(column.Value(j))
					case *array.Uint64:
						builder.Field(colIdx).(*array.Uint64Builder).Append(column.Value(j))
					case *array.Int8:
						builder.Field(colIdx).(*array.Int8Builder).Append(column.Value(j))
					case *array.Int16:
						builder.Field(colIdx).(*array.Int16Builder).Append(column.Value(j))
					case *array.Int32:
						builder.Field(colIdx).(*array.Int32Builder).Append(column.Value(j))
					case *array.Int64:
						builder.Field(colIdx).(*array.Int64Builder).Append(column.Value(j))
					case *array.Float16:
						builder.Field(colIdx).(*array.Float16Builder).Append(column.Value(j))
					case *array.Float32:
						builder.Field(colIdx).(*array.Float32Builder).Append(column.Value(j))
					case *array.Float64:
						builder.Field(colIdx).(*array.Float64Builder).Append(column.Value(j))
					default:
						log.Fatalf("Unsupported column type: %T", column)
					}
				}
			}
		}

		record.Release()

		// We have to count the rows we will eventually write
		// not the amount we've written
		// since the chunk size may not align neatly with the number of requested loci
		// If the channel is closed due to the count
		// we will clean up and write the remaining chunk before exiting
		count.Add(int64(rowsAccepted))

		if rowsAccumulated >= WRITE_CHUNK_SIZE {
			// Create a new record from the row
			filteredRecord := builder.NewRecord()

			// Write the new record to the output file
			if err := arrowWriter.WriteChunk(filteredRecord); err != nil {
				log.Fatal(err)
			}

			filteredRecord.Release()
			builder.Release()

			builder = array.NewRecordBuilder(pool, arrowWriter.Schema)

			rowsAccumulated = 0
		}
	}

	if rowsAccumulated >= 0 {
		filteredRecord := builder.NewRecord()

		if err := arrowWriter.WriteChunk(filteredRecord); err != nil {
			log.Fatal(err)
		}

		filteredRecord.Release()
		builder.Release()
		rowsAccumulated = 0
	}

	complete <- true
}

func main() {
	var totalCount atomic.Int64

	cliargs := setup(nil)

	loci, err := readLociFile(cliargs.lociPath)
	if err != nil {
		log.Fatalf("Could not read loci file: %v", err)
	}

	// Open Arrow IPC file
	file, err := os.Open(cliargs.inputPath)
	if err != nil {
		log.Fatalf("Failed to open IPC file: %v", err)
	}
	defer file.Close()

	pool := memory.NewGoAllocator()
	fr, err := ipc.NewFileReader(file, ipc.WithAllocator(pool))
	if err != nil {
		log.Fatalf("Failed to create IPC file reader: %v", err)
	}

	schema := fr.Schema()

	outFile, err := os.Create(cliargs.outputPath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	arrowWriter, err := bystroArrow.NewArrowIPCFileWriterWithSchema(outFile, schema, ipc.WithZstd())
	if err != nil {
		log.Fatal(err)
	}
	defer arrowWriter.Close()

	totalRecords := fr.NumRecords()
	numWorkers := runtime.NumCPU() * 2

	// var wg sync.WaitGroup
	workQueue := make(chan int, 16)
	complete := make(chan bool)

	// Spawn threads
	for i := 0; i < numWorkers; i++ {
		go processRecordAt(fr, loci, arrowWriter, workQueue, complete, &totalCount)
	}

	progressSender, err := beanstalkd.CreateMessageSender(cliargs.beanstalkConfigPath, cliargs.jobSubmissionID, cliargs.noBeanstalkd)
	if err != nil {
		log.Fatalf("Couldn't create message sender due to: [%s]\n", err)
	}

	var progressUpdate int64 = 0
	var totalRowsProcessed int64 = 0

	checkInteval := totalRecords / 100
	totalRows := int64(len(loci))
	totalRowsProcessed = 0

	var hasClosed bool
	for i := 0; i < totalRecords; i++ {
		workQueue <- i

		if i%checkInteval == 0 {
			totalRowsProcessed = totalCount.Load()

			if totalRowsProcessed >= totalRows {
				fmt.Printf("Finished processing all loci by record %d\n", i)

				close(workQueue)
				hasClosed = true

				break
			}

			progressUpdate += totalRowsProcessed

			if progressUpdate >= cliargs.progressFrequency {
				progressSender.SetProgress(int(totalRowsProcessed))
				go progressSender.SendMessage()
				progressUpdate = 0
			}
		}
	}

	if !hasClosed {
		close(workQueue)
	}

	for i := 0; i < numWorkers; i++ {
		<-complete
	}
}
