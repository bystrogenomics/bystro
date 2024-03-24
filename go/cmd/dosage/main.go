package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/apache/arrow/go/v14/arrow/arrio"
	"github.com/apache/arrow/go/v14/arrow/ipc"
	"github.com/apache/arrow/go/v14/arrow/memory"
)

func main() {
	// Accept an output arg that is a string, and a variadic number of input files
	var (
		outputArg string
		inputArgs []string
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "dosage --output <output_file_path> <dosage_matrix1> <dosage_matrix2> <dosage_matrixN>\n")
	}

	flag.StringVar(&outputArg, "output", "", "output arg that is a string")
	flag.Parse()

	inputArgs = flag.Args()

	err := processFiles(outputArg, inputArgs)
	if err != nil {
		log.Fatal(err)
	}
}

func processFile(ipcFileWriter *ipc.FileWriter, filePath string, pool memory.Allocator) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("could not open file %s: %w", filePath, err)
	}
	defer file.Close()

	r, err := ipc.NewFileReader(file, ipc.WithAllocator(pool))
	if err != nil {
		return fmt.Errorf("could not create file reader for %s: %w", filePath, err)
	}
	defer r.Close()

	n, err := arrio.Copy(ipcFileWriter, r)
	if err != nil {
		return fmt.Errorf("could not copy ARROW stream: %w", err)
	}
	if got, want := n, int64(r.NumRecords()); got != want {
		return fmt.Errorf("invalid number of records written (got=%d, want=%d)", got, want)
	}

	return nil
}

func processFiles(outPath string, inPaths []string) error {
	dest, err := os.Create(outPath)
	if err != nil {
		log.Fatal(err)
	}
	defer dest.Close()

	mem := memory.NewGoAllocator()

	// Read the first file and get the schema, then write that file
	r, err := os.Open(inPaths[0])
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()

	rr, err := ipc.NewFileReader(r, ipc.WithAllocator(mem))
	if err != nil {
		if errors.Is(err, io.EOF) {
			return nil
		}
		return err
	}
	defer rr.Close()

	ww, err := ipc.NewFileWriter(dest, []ipc.Option{ipc.WithAllocator(mem), ipc.WithSchema(rr.Schema()), ipc.WithZstd()}...)
	if err != nil {
		return fmt.Errorf("could not create ARROW file writer: %w", err)
	}

	defer ww.Close()

	n, err := arrio.Copy(ww, rr)
	if err != nil {
		return fmt.Errorf("could not copy ARROW stream: %w", err)
	}
	if got, want := n, int64(rr.NumRecords()); got != want {
		return fmt.Errorf("invalid number of records written (got=%d, want=%d)", got, want)
	}

	// Read the rest of the files and append them
	for _, inPath := range inPaths[1:] {
		err := processFile(ww, inPath, mem)

		if err != nil {
			return err
		}
	}

	err = ww.Close()
	if err != nil {
		return fmt.Errorf("could not close output ARROW stream: %w", err)
	}

	return nil
}
