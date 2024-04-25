package main

import (
	"bufio"
	"bystro/beanstalkd"

	"github.com/tidwall/btree"

	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"time"

	"github.com/apache/arrow/go/v14/arrow/array"
	"github.com/apache/arrow/go/v14/arrow/ipc"
	"github.com/apache/arrow/go/v14/arrow/memory"
	bystroArrow "github.com/bystrogenomics/bystro-vcf/arrow"
)

var WRITE_CHUNK_SIZE = 5000

type void struct{}

var member void

type CLIArgs struct {
	inputPath           string
	outputPath          string
	beanstalkConfigPath string
	jobSubmissionID     string
	lociPath            string
	progressFrequency   int64
}

// setup parses the command-line arguments and returns a CLIArgs struct.
func setup(args []string) *CLIArgs {
	cliargs := &CLIArgs{}
	fs := flag.NewFlagSet("CLIArgs", flag.ExitOnError)

	fs.StringVar(&cliargs.inputPath, "input", "", "The path to the input Arrow IPC/Feather file")
	fs.StringVar(&cliargs.inputPath, "in", "", "The path to the input Arrow IPC/Feather file")
	fs.StringVar(&cliargs.lociPath, "loci", "", "The path to a file containing loci to filter, one locus per line")
	fs.StringVar(&cliargs.outputPath, "output", "", "The path to the output Arrow IPC/Feather file")
	fs.StringVar(&cliargs.outputPath, "out", "", "The path to the output Arrow IPC/Feather file (alias)")

	fs.StringVar(&cliargs.beanstalkConfigPath, "queue-config", "", "The path to the Beanstalkd queue connection config (e.g., config/beanstalk.yml)")
	fs.StringVar(&cliargs.beanstalkConfigPath, "q", "", "The path to the Beanstalkd queue connection config (short form)")

	fs.Int64Var(&cliargs.progressFrequency, "progress-frequency", int64(5e3), "Print progress every N variants processed")
	fs.Int64Var(&cliargs.progressFrequency, "p", int64(5e3), "Print progress every N variants processed (short form)")

	fs.StringVar(&cliargs.jobSubmissionID, "job-submission-id", "", "The job submission ID")
	fs.StringVar(&cliargs.jobSubmissionID, "j", "", "The job submission ID (short form)")

	// Parsing custom args or os.Args based on context
	a := os.Args[1:]
	if args != nil {
		a = args
	}

	fs.Parse(a)

	if err := validateArgs(cliargs); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		fs.Usage()
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

// readLociFile reads a file containing loci and stores them in a map with their hash values as keys.
func readLociFile(filePath string) (*btree.Set[string], error) {
	var loci btree.Set[string]

	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		loci.Insert(scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return &loci, nil
}

// processRecordAt processes a record at the given index, filters the rows based on the loci,
// and writes the filtered rows to the arrowWriter.
func processRecordAt(fr *ipc.FileReader, loci *btree.Set[string], arrowWriter *bystroArrow.ArrowWriter, queue chan int, complete chan bool, count *atomic.Int64) {
	pool := memory.NewGoAllocator()
	builder := array.NewRecordBuilder(pool, arrowWriter.Schema)
	defer builder.Release()

	rowsAccumulated := 0

	for index := range queue {
		record, err := fr.RecordAt(index)
		if err != nil {
			log.Fatalf("Failed to read record at %d: %v", index, err)
		}

		locusCol := record.Column(0).(*array.String)

		rowsAccepted := 0
		for j := 0; j < int(record.NumRows()); j++ {
			locusValue := locusCol.Value(j)
			if loci.Contains(locusValue) {
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

	// Write the remaining rows if the loop is terminated early
	// TODO 2024-04-24 @akotlar confirm whether this is necessary
	if rowsAccumulated > 0 {
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
	runProgram(nil)
}

func runProgram(args []string) {
	var totalCount atomic.Int64

	cliargs := setup(args)

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	log.Printf("Memory usage before reading loci file: %d MB\n", m.Alloc/1024/1024)

	loci, err := readLociFile(cliargs.lociPath)
	if err != nil {
		log.Fatalf("Could not read loci file: %v", err)
	}

	// Get memory usage after reading loci file:
	runtime.ReadMemStats(&m)
	log.Printf("Memory usage after reading loci file: %d MB\n", m.Alloc/1024/1024)

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

	progressSender, err := beanstalkd.CreateMessageSender(cliargs.beanstalkConfigPath, cliargs.jobSubmissionID, "saveFromQuery")
	if err != nil {
		log.Fatalf("Couldn't create message sender due to: [%s]\n", err)
	}

	totalRows := int64(loci.Len())
	quit := make(chan bool)
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		var lastUpdate int64
		var currentCount int64
		for {
			select {
			case <-quit:
				log.Println("Progress goroutine exiting...")
				return

			case <-ticker.C:
				// Send your message here. For demonstration, we'll just print.
				currentCount = totalCount.Load()

				if currentCount-lastUpdate >= cliargs.progressFrequency {
					lastUpdate = currentCount
					progressSender.SendStringMessage(fmt.Sprintf("Dosage: Filtered %d of %d variants", int(lastUpdate), totalRows))
				}
			}
		}
	}()

	for i := 0; i < totalRecords; i++ {
		workQueue <- i

		if totalCount.Load() >= totalRows {
			break
		}
	}

	close(workQueue)

	for i := 0; i < numWorkers; i++ {
		<-complete
	}

	quit <- true

	progressSender.SendStringMessage(fmt.Sprintf("Dosage: filtered %d of %d variants", int(totalCount.Load()), totalRows))
}
