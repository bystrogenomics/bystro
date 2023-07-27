#Extract gnomad loadings variant list from 1kgp genomes using plink

#TODO convert this process to one using our own vcf parser

#Download plink2 - can be downloaded with example below or at https://www.cog-genomics.org/plink/2.0/
#wget https://s3.amazonaws.com/plink2-assets/plink2_linux_avx2_20230707.zip
#unzip plink2_linux_avx2_20230707.zip 

#Download 1kgp genomes - make sure this is the most recent version
wget 'ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr'*

#Gnomad loadings have been preprocessed to extract the variant list only as gnomadvariantlist.txt

#Extract from each autosomal chromosome for ancestry
for ((chr=1; chr<=22; chr++))
do
    #Input and output paths for each chromosome
    input_vcf="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    output_base="chr${chr}"

    #Run plink2 with current chromosome
    ./plink2 --extract gnomadvariantlist.txt \
             --make-pgen \
             --out "$output_base" \
             --vcf "$input_vcf"
     # Append output file name to the merge list, excluding 'chr1'
    if [ "$chr" -ne 1 ]; then
        echo "$output_base" >> chr_merge_list.txt
    fi

done

#Merge the files together using list of output file names
./plink2 --export vcf \
         --out 1kgpGnomadList \
         --pfile chr1 \
         --pmerge-list chr_merge_list.txt