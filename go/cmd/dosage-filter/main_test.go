package main

import (
	"os"
	"path"
	"path/filepath"
	"testing"

	"github.com/apache/arrow/go/v14/arrow"
	"github.com/apache/arrow/go/v14/arrow/array"
	"github.com/apache/arrow/go/v14/arrow/ipc"
	bystroArrow "github.com/bystrogenomics/bystro-vcf/arrow"
)

// TestValidateArgsMissingRequired tests the behavior of validateArgs when required arguments are missing
func TestValidateArgsMissingRequired(t *testing.T) {
	tests := []struct {
		name    string
		args    CLIArgs
		wantErr bool
	}{
		{
			name: "All arguments provided",
			args: CLIArgs{
				inputPath:           "input.feather",
				outputPath:          "output.feather",
				lociPath:            "loci.txt",
				beanstalkConfigPath: "config.yml",
				jobSubmissionID:     "123",
			},
			wantErr: false,
		},
		{
			name: "Missing inputPath",
			args: CLIArgs{
				outputPath:          "output.feather",
				lociPath:            "loci.txt",
				beanstalkConfigPath: "config.yml",
				jobSubmissionID:     "123",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateArgs(&tt.args); (err != nil) != tt.wantErr {
				t.Errorf("validateArgs() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func createArrowFile(filePath string, fieldTypes []arrow.DataType, fieldNames []string, rows [][]any) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}

	writer, err := bystroArrow.NewArrowIPCFileWriter(file, fieldNames, fieldTypes)
	if err != nil {
		return err
	}

	builder, err := bystroArrow.NewArrowRowBuilder(writer, len(rows))
	if err != nil {
		return err
	}

	for _, row := range rows {
		if err := builder.WriteRow(row); err != nil {
			return err
		}
	}

	if err := builder.Release(); err != nil {
		return err
	}

	if err := writer.Close(); err != nil {
		return err
	}

	return err
}

func TestArrowWriteRead(t *testing.T) {
	filePath := path.Join(t.TempDir(), "test.feather")

	// Define the data types for the fields
	fieldTypes := []arrow.DataType{arrow.BinaryTypes.String, arrow.PrimitiveTypes.Uint8, arrow.PrimitiveTypes.Uint8}
	fieldNames := []string{"Locus", "Sample1", "Sample2"}
	rows := make([][]any, 3)
	rows[0] = []any{"chr1:1000:A:T", uint8(0), uint8(1)}
	rows[1] = []any{"chr2:1000:A:T", uint8(0), uint8(1)}
	rows[2] = []any{"chr3:1000:A:T", uint8(10), uint8(12)}

	err := createArrowFile(filePath, fieldTypes, fieldNames, rows)
	if err != nil {
		t.Fatal(err)
	}

	// Read the file
	file, err := os.Open(filePath)
	if err != nil {
		t.Fatal(err)
	}

	// write loci file to temp dir
	lociFilePath := filepath.Join(t.TempDir(), "loci.txt")
	lociFile, err := os.Create(lociFilePath)
	if err != nil {
		t.Fatal(err)
	}

	lociFile.WriteString("chr1:1000:A:T\n")
	lociFile.WriteString("chr3:1000:A:T\n")

	lociFile.Close()

	outFilePath := path.Join(t.TempDir(), "output.feather")

	// Call main
	args := []string{"-in", filePath, "-out", outFilePath, "-loci", lociFilePath}
	runProgram(args)

	// Read outFilePath and ensure it is missing chr2 locus entry
	file, err = os.Open(outFilePath)
	if err != nil {
		t.Error(err)
	}

	reader, err := ipc.NewFileReader(file)
	if err != nil {
		t.Error(err)
	}
	defer reader.Close()

	if reader.NumRecords() != 1 {
		t.Error("Expected 1 record, got ", reader.NumRecords())
	}

	record, err := reader.Record(0)
	if err != nil {
		t.Error(err)
	}

	columns := record.Columns()
	if len(columns) != 3 {
		t.Error("Expected 3 columns")
	}

	if columns[0].Len() != 2 {
		t.Errorf("Expected 2 rows (1 filtered), got %d", columns[0].Len())
	}

	if columns[1].Len() != 2 {
		t.Errorf("Expected 2 rows (1 filtered), got %d", columns[1].Len())
	}

	if columns[2].Len() != 2 {
		t.Errorf("Expected 2 rows (1 filtered), got %d", columns[2].Len())
	}

	// Read the first row
	locus1 := columns[0].(*array.String).Value(0)
	sample1 := columns[1].(*array.Uint8).Value(0)
	sample2 := columns[2].(*array.Uint8).Value(0)

	if locus1 != "chr1:1000:A:T" {
		t.Errorf("Expected chr1:1000:A:T, got %s", locus1)
	}

	if sample1 != 0 {
		t.Errorf("Expected 0, got %d", sample1)
	}

	if sample2 != 1 {
		t.Errorf("Expected 1, got %d", sample2)
	}

	// Read the second row
	locus2 := columns[0].(*array.String).Value(1)
	sample1 = columns[1].(*array.Uint8).Value(1)
	sample2 = columns[2].(*array.Uint8).Value(1)

	if locus2 != "chr3:1000:A:T" {
		t.Errorf("Expected chr3:1000:A:T, got %s", locus2)
	}

	if sample1 != 10 {
		t.Errorf("Expected 10, got %d", sample1)
	}

	if sample2 != 12 {
		t.Errorf("Expected 12, got %d", sample2)
	}

}

// test 0 loci retained
func TestArrowWriteRead0Loci(t *testing.T) {
	filePath := path.Join(t.TempDir(), "test0.feather")

	// Define the data types for the fields
	fieldTypes := []arrow.DataType{arrow.BinaryTypes.String, arrow.PrimitiveTypes.Uint8, arrow.PrimitiveTypes.Uint8}
	fieldNames := []string{"Locus", "Sample1", "Sample2"}
	rows := make([][]any, 3)
	rows[0] = []any{"chr1:1000:A:T", uint8(0), uint8(1)}
	rows[1] = []any{"chr2:1000:A:T", uint8(0), uint8(1)}
	rows[2] = []any{"chr3:1000:A:T", uint8(10), uint8(12)}

	err := createArrowFile(filePath, fieldTypes, fieldNames, rows)
	if err != nil {
		t.Fatal(err)
	}

	// write loci file to temp dir
	lociFilePath := filepath.Join(t.TempDir(), "loci0.txt")
	lociFile, err := os.Create(lociFilePath)
	if err != nil {
		t.Fatal(err)
	}

	lociFile.Close()

	outFilePath := filepath.Join("output0.feather")

	args := []string{"-in", filePath, "-out", outFilePath, "-loci", lociFilePath}
	runProgram(args)

	// assert that the outputFile exists
	if _, err := os.Stat(outFilePath); os.IsNotExist(err) {
		t.Errorf("Expected output file to exist, got %v", err)
	}

	// Read outFilePath and ensure it is missing chr2 locus entry
	file, err := os.Open(outFilePath)
	if err != nil {
		t.Error(err)
	}

	reader, err := ipc.NewFileReader(file)
	if err != nil {
		t.Error(err)
	}
	defer reader.Close()

	if reader.NumRecords() != 0 {
		t.Error("Expected 0 records, got ", reader.NumRecords())
	}
}
