#!/bin/env bash

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]..."
    echo "Convert a custom file format to VCF."
    echo "Reads from standard input if '-' is provided as an argument, or use --input to specify a file."
    echo
    echo "  -h, --help    Display this help and exit"
    echo "  --input FILE  Specify the input file to be converted"
}

# No arguments provided (not even a dash)
if [ "$#" -eq 0 ]; then
    show_help
    exit 1
fi

# Variables
input_file=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --input)
            if [[ -n $input_file ]]; then
                echo "Error: Cannot use --input with '-' argument."
                exit 1
            fi
            input_file="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        -)
            if [[ -n $input_file ]]; then
                echo "Error: Cannot use '-' with --input argument."
                exit 1
            fi
            input_file="-" # Indicate that we should read from stdin
            shift
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done


# Function to extract the reference genome, CADD version, and check headers
check_headers_and_extract_details() {
    local input=$1
    local line_no=0
    local ref_genome=""
    local cadd_version=""

    cadd_header_regex='^## CADD (GRCh[0-9]+)-v([0-9.]+)'
    cadd_header_line_two="^#Chrom\s+Pos\s+Ref\s+Alt\s+RawScore\s+PHRED"

    while IFS= read -r line; do
        ((line_no++))
        if [ $line_no -eq 1 ]; then
            if [[ $line =~ $cadd_header_regex ]]; then
                ref_genome=${BASH_REMATCH[1]}
                cadd_version=${BASH_REMATCH[2]}
            else
                echo "Error: Invalid header line or reference genome and CADD version not found." >&2
                exit 1
            fi
        elif [ $line_no -eq 2 ]; then
            if ! [[ "$line" =~ $cadd_header_line_two ]]; then
                echo "Error: Second header line does not match expected format." >&2
                exit 1
            fi
            break
        fi
    done < "$input"

    echo "$ref_genome $cadd_version"
}

# Check input and extract reference genome and CADD version
if [[ $input_file == "-" ]]; then
    read ref_genome cadd_version <<< $(check_headers_and_extract_details /dev/stdin)
else
    if [[ ! -f $input_file ]]; then
        echo "Error: File does not exist." >&2
        exit 1
    fi
    read ref_genome cadd_version <<< $(check_headers_and_extract_details "$input_file")
fi

# Check if the ref_genome and cadd_version were properly extracted
if [ -z "$ref_genome" ] || [ -z "$cadd_version" ]; then
    echo "Error: Unable to extract reference genome and CADD version." >&2
    exit 1
fi

# Output VCF Header
cat <<EOF
##fileformat=VCFv4.2
##source=CADD_${ref_genome}-v${cadd_version}
##reference=${ref_genome}
##INFO=<ID=RawScore,Number=1,Type=Float,Description="Raw CADD score">
##INFO=<ID=PHRED,Number=1,Type=Float,Description="CADD PHRED-like score">
EOF
echo -e "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"

# Process the file or stdin and convert it to VCF format using awk
if [[ $input_file == "-" ]]; then
    # Read from stdin
    awk 'FNR > 2 { printf("%s\t%s\t.\t%s\t%s\t.\t.\tRawScore=%s;PHRED=%s\n", $1, $2, $3, $4, $5, $6) }'
else
    # Check if the file exists
    if [[ ! -f $input_file ]]; then
        echo "Error: File does not exist."
        exit 1
    fi
    awk 'FNR > 2 { printf("%s\t%s\t.\t%s\t%s\t.\t.\tRawScore=%s;PHRED=%s\n", $1, $2, $3, $4, $5, $6) }' "$input_file"
fi