#!/usr/bin/env bash
set -e

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo -e "\nUsage: $0 <hg19_directory> <hg38_directory>\n"
    exit 1
fi

# Get directories from arguments and strip trailing slashes
hg19_dir="${1%/}"
hg38_dir="${2%/}"

# Assembly reports for renaming chromosomes
report_dir='ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405'
wget -N "${report_dir}/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_assembly_report.txt"
wget -N "${report_dir}/GCF_000001405.25_GRCh37.p13/GCF_000001405.25_GRCh37.p13_assembly_report.txt"

# Grab the useful columns
for k in *assembly_report.txt
  do
    out=$(echo $k | sed 's/.txt/.chrnames/')
    grep -e '^[^#]' $k | awk '{ print $7, $1 }' > $out
done

bcftools annotate \
  --rename-chrs GCF_000001405.25_GRCh37.p13_assembly_report.chrnames \
  --threads 10 -Oz -o ${hg19_dir}/GRCh37.dbSNP155.vcf.gz ${hg19_dir}/GCF_000001405.25.gz
bcftools annotate \
  --rename-chrs GCF_000001405.39_GRCh38.p13_assembly_report.chrnames \
  --threads 10 -Oz -o ${hg38_dir}/GRCh38.dbSNP155.vcf.gz ${hg38_dir}/GCF_000001405.39.gz