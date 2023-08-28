#Extract gnomad loadings variant list from 1kgp genomes using plink

#TODO convert this process to one using our own vcf parser

# Plink2 is needed for this pre-process
# If installed in current directory, add to path
export PATH=$PATH:./
# Check if plink2 is installed or not in path
program_name="plink2"
if ! command -v "$program_name" &> /dev/null; then
    echo "Error: '$program_name' is not installed or not present on system PATH."
    echo "Please install '$program_name' or make sure it is added to the PATH.  You can install from: https://www.cog-genomics.org/plink/2.0/"
    exit 1
fi
echo "'$program_name' is installed and present on the system's PATH."

# Download 1kgp manifest that has list of vcf files with checksums - make sure this is the most recent version of 1kgp
wget 'ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/20220804_manifest.txt'
# Download 1kgp genomes 
wget 'ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr'*

# Check if downloaded files were properly downloaded using md5 checksums
MANIFEST_FILE="20220804_manifest.txt"
# Variable to keep track of the checksum verification status
verification_status="PASSED"
# Verify checksums for each file in the manifest and print pass/fail
while read -r file_name expected_checksum; do
    # Calculate md5sum checksum of the downloaded file
    actual_checksum=$(md5sum "$file_name" | awk '{print $1}')

    # Compare the calculated checksum with the expected checksum and mark if failure
    if [ "$actual_checksum" != "$expected_checksum" ]; then
        verification_status="FAILED"
        echo "Checksum verification FAILED for $file_name"
    fi
done < "$MANIFEST_FILE"

if [ "$verification_status" = "PASSED" ]; then
    echo "All checksum verifications PASSED"
else
    echo "One or more checksum verifications FAILED"
fi

# Gnomad loadings have been preprocessed to extract the variant list only as gnomadvariantlist.txt

# Extract from each autosomal chromosome for ancestry
for ((chr=1; chr<=22; chr++))
do
    # Input and output paths for each chromosome
    input_vcf="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    output_base="tempchr${chr}"

    # Run plink2 with current chromosome
    ./plink2 --extract gnomadvariantlist.txt \
             --make-pgen \
             --out "$output_base" \
             --vcf "$input_vcf"
     # Append output file name to the merge list, excluding 'chr1'
    if [ "$chr" -ne 1 ]; then
        echo "$output_base" >> chr_merge_list.txt
    fi
done

# Merge the files together using list of output file names
./plink2 --export vcf \
         --out 1kgpGnomadList \
         --pfile tempchr1 \
         --pmerge-list chr_merge_list.txt
         
#Delete temp files that are no longer needed
echo "Clean up temp files"
rm "1kGP_high_coverage_Illumina.chr"*
rm "tempchr"*
rm "chr_merge_list.txt"
rm "$MANIFEST_FILE"