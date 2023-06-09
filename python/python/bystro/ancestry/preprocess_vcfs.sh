#!/bin/bash

set -e

FTP_PREFIX="ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"

if [[ $(ls chr*_filtered.vcf.gz) ]]
then
   echo "Found temp files of form: chr*_filtered.vcf.gz, please remove before continuing"
   exit 1
fi
	
for chr in {1..22};do
    echo "Processing chromosome $chr at" `date`
    vcf_filename=ALL.chr$chr.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
    #vcf_filename=chr$chr.vcf.gz
    if [ ! -f $vcf_filename ]
    then
    	echo "didn't find $vcf_filename in directory; downloading"
    	wget $FTP_PREFIX$vcf_filename
    else
    	echo "Found $vcf_filename in directory"
    fi

    
    output_filename="chr${chr}_filtered.vcf.gz"
    echo "writing to temp file:" $output_filename
    bcftools view -I $vcf_filename -Ou |   # exclude indels
    bcftools view -i 'MAF > 0.01' -Ou |  # exclude  MAF < 0.01
    bcftools view -c 1 -Ou |  # exclude monomorphic sites
    bcftools norm -d all -Ou | # include only bi-allelic sites
    bcftools +prune -l 0.2 -w 1000 | # last output must be human-readable
    gzip -c > $output_filename
done

final_outfile=1KGP_final_variants_1percent.vcf.gz

echo "Writing final variants to: " $final_outfile
vcf-concat chr*_filtered.vcf.gz | gzip -c > $final_outfile

echo "Cleaning up"
rm chr*_filtered.vcf.gz
