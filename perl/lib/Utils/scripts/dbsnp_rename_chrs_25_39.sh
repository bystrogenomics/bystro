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
# The structure of the assembly report is:
# Sequence-Name Sequence-Role   Assigned-Molecule       Assigned-Molecule-Location/Type GenBank-Accn    Relationship    RefSeq-Accn     Assembly-Unit   Sequence-Length UCSC-style-name
# 1       assembled-molecule      1       Chromosome      CM000663.2      =       NC_000001.11    Primary Assembly        248956422       chr1
# 2       assembled-molecule      2       Chromosome      CM000664.2      =       NC_000002.12    Primary Assembly        242193529       chr2
# 3       assembled-molecule      3       Chromosome      CM000665.2      =       NC_000003.12    Primary Assembly        198295559       chr3
# 4       assembled-molecule      4       Chromosome      CM000666.2      =       NC_000004.12    Primary Assembly        190214555       chr4
# 5       assembled-molecule      5       Chromosome      CM000667.2      =       NC_000005.10    Primary Assembly        181538259       chr5
# 6       assembled-molecule      6       Chromosome      CM000668.2      =       NC_000006.12    Primary Assembly        170805979       chr6
# 7       assembled-molecule      7       Chromosome      CM000669.2      =       NC_000007.14    Primary Assembly        159345973       chr7
# 8       assembled-molecule      8       Chromosome      CM000670.2      =       NC_000008.11    Primary Assembly        145138636       chr8
# 9       assembled-molecule      9       Chromosome      CM000671.2      =       NC_000009.12    Primary Assembly        138394717       chr9
# 10      assembled-molecule      10      Chromosome      CM000672.2      =       NC_000010.11    Primary Assembly        133797422       chr10
# 11      assembled-molecule      11      Chromosome      CM000673.2      =       NC_000011.10    Primary Assembly        135086622       chr11
# 12      assembled-molecule      12      Chromosome      CM000674.2      =       NC_000012.12    Primary Assembly        133275309       chr12
# 13      assembled-molecule      13      Chromosome      CM000675.2      =       NC_000013.11    Primary Assembly        114364328       chr13
# 14      assembled-molecule      14      Chromosome      CM000676.2      =       NC_000014.9     Primary Assembly        107043718       chr14
# 15      assembled-molecule      15      Chromosome      CM000677.2      =       NC_000015.10    Primary Assembly        101991189       chr15
# 16      assembled-molecule      16      Chromosome      CM000678.2      =       NC_000016.10    Primary Assembly        90338345        chr16
# 17      assembled-molecule      17      Chromosome      CM000679.2      =       NC_000017.11    Primary Assembly        83257441        chr17
# 18      assembled-molecule      18      Chromosome      CM000680.2      =       NC_000018.10    Primary Assembly        80373285        chr18
# 19      assembled-molecule      19      Chromosome      CM000681.2      =       NC_000019.10    Primary Assembly        58617616        chr19
# 20      assembled-molecule      20      Chromosome      CM000682.2      =       NC_000020.11    Primary Assembly        64444167        chr20
# 21      assembled-molecule      21      Chromosome      CM000683.2      =       NC_000021.9     Primary Assembly        46709983        chr21
# 22      assembled-molecule      22      Chromosome      CM000684.2      =       NC_000022.11    Primary Assembly        50818468        chr22
# X       assembled-molecule      X       Chromosome      CM000685.2      =       NC_000023.11    Primary Assembly        156040895       chrX
# Y       assembled-molecule      Y       Chromosome      CM000686.2      =       NC_000024.10    Primary Assembly        57227415        chrY
# MT      assembled-molecule      MT      Chromosome      J01415.2        =       NC_012920.1     Primary Assembly        16569           chrM

# The 11th column contains UCSC-style contigs. Unfortunately, when using the UCSC-style-name column
# not all contigs get values. For example, the contig "KI270728.1" is not present in the UCSC-style-name, for example
# HG1459_PATCH    fix-patch       X       Chromosome      JH806600.2      =       NW_004070890.2  PATCHES 6530008 na
# Therefore, to get a valid VCF we need to use the Sequence-Name column
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