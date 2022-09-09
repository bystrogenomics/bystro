#Workflow for Admixture Ancestry Scoring

#Testing admixture locally with Hapmap data in this path
export PATH=$PATH:/Applications/GenomicsTools
admixture hapmap3.bed 3

# Verify the two datasets have the same set of SNPs
diff -s 1kGPreference.bim study.bim
# Run unsupervised ADMIXTURE with K=2
admixture 1kGPreference.bed 2
# Use learned allele frequencies as (fixed) input to next step
cp 1kGPreference.2.P study.2.P.in
# Run projection ADMIXTURE with K=2
admixture -P study.bed 2

#Using 1kgenomes as ref - http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/
#Before using ref, converted vcf to plink2 files and merge

#common snps only for PCA using downloaded bystro file (maf>.05)
cut -f1,2,5,65,66 Annotation/SamplePopcommonsnps.annotation.tsv > Annotation/PartialCommon.txt
python ConvertChrPosToRsCommonsnps.py Annotation/PartialCommon.txt Annotation/CommonWchrRs.txt
cut -f4 Annotation/CommonWchrRs.txt > $SnpPath/commonSnpListforgwas.txt
#need to clean plink file before running pca, then extract common snps
plink2 --bfile $SnpPath/CleanPlinkFile --extract $SnpPath/commonSnpListforgwas.txt --allow-extra-chr --make-bed --out $SnpPath/FileForpca
# Identify LD between SNPs
plink2 --bfile $SnpPath/FileForpca --indep-pairwise 50 5 0.2 --allow-extra-chr --out $SnpPath/allcleanLD
# LD pruning
plink2 --bfile $SnpPath/FileForpca --extract $SnpPath/allcleanLD.prune.out --allow-extra-chr --make-bed -out $SnpPath/pruned
###PCA 
plink2 --bfile $SnpPath/pruned --allow-extra-chr --pca 10
