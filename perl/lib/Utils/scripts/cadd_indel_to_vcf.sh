#!/bin/bash

# Check for the input file
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 input_file"
    exit 1
fi

input_file=$1

# VCF Header
echo "##fileformat=VCFv4.2"
echo "##source=CADD_GRCh37-v1.6"
echo "##reference=GRCh37"
echo '##INFO=<ID=RawScore,Number=1,Type=Float,Description="Raw CADD score">'
echo '##INFO=<ID=PHRED,Number=1,Type=Float,Description="CADD PHRED-like score">'
echo -e "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"

# Process the input file and convert it to VCF format
tail -n +3 "$input_file" | while read -r line; do
    # Read the columns into variables
    read -a fields <<< "$line"
    chrom=${fields[0]}
    pos=${fields[1]}
    ref=${fields[2]}
    alt=${fields[3]}
    rawscore=${fields[4]}
    phred=${fields[5]}

    # Create the INFO field content
    info="RawScore=${rawscore};PHRED=${phred}"

    # VCF columns: CHROM POS ID REF ALT QUAL FILTER INFO
    echo -e "${chrom}\t${pos}\t.\t${ref}\t${alt}\t.\t.\t${info}"

done
